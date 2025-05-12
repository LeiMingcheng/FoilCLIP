#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLIP 图文对齐训练脚本（包含 CST 参数回归）- 适配H5数据集格式
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import argparse
import numpy as np
import h5py
import json
from tqdm import tqdm
import torch.distributed.nn as distnn
from Model import (
    TextCSTPairH5Dataset, text_cst_collate_fn, TextCSTAlignmentModel,
    SoftContrastiveLoss, loss_function, contrastive_loss
)

def train_with_validation(model, train_dataloader, val_dataloader, device, num_epochs=10, 
                         print_interval=40, checkpoint_dir="./checkpoint",
                         lambda_recon=1.0, lambda_kl=0.001, lambda_align=1.0, 
                         lambda_phy=0.5, alpha=0.5, temperature=0.07, stage=None, args=None):
    """
    带有验证集的训练函数。

    参数:
        model: 训练的模型。
        train_dataloader: 训练数据加载器。
        val_dataloader: 验证数据加载器。
        device: 训练设备 (例如, "cuda" 或 "cpu")。
        num_epochs: 训练的总轮数。
        print_interval: 打印训练信息的间隔步数。
        checkpoint_dir: 保存模型检查点的目录。
        lambda_recon: 重构损失的权重。
        lambda_kl: KL散度损失的权重。
        lambda_align: 对齐损失的权重。
        lambda_phy: 物理维度约束损失的权重。
        alpha: 粗粒度对齐损失的权重。
        temperature: 对比学习的温度系数。
        stage: 当前训练阶段的标识 (例如, "stage1", "stage2")。
        args: 命令行参数。
    """
    best_loss = float('inf') # 初始化最佳损失为正无穷大

    # 仅在主进程创建检查点目录
    if args.local_rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"创建检查点目录: {checkpoint_dir}")
    
    stage_info = f"[{stage}] " if stage else "" # 日志中阶段信息的前缀
    # 温度参数的初始值和最终值
    
    for epoch in range(num_epochs):
        # 为分布式训练设置采样器的epoch
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'set_epoch'):
            val_dataloader.sampler.set_epoch(epoch)
        
        # 根据训练阶段动态调整物理对齐权重和alpha值
        if stage == "stage2":
            # 训练进度比例
            progress_ratio = epoch / num_epochs
            # 当前物理权重，随训练进度增加
            current_phy_lambda =  lambda_phy *(0.1 + min(1.0, progress_ratio * 2))
            
            # 当前alpha值，从2递减到0.5
            current_alpha = (2.0 - (1.5 * min(1.0, progress_ratio * 2))) * alpha
            # 获取当前温度参数
            if args.use_variable_temperature == 1:
                temp_start = 0.045 # 温度起始值
                temp_end = 0.18   # 温度结束值
                # 当前温度，随训练进度增加
                current_temperature = temp_start + (temp_end - temp_start) * min(1.0, progress_ratio * 1.5)
            else:
                current_temperature = temperature # 固定温度
            phy_desc = f", 物理权重:{current_phy_lambda:.4f}, alpha:{current_alpha:.2f}, temp:{current_temperature:.4f}"
        else:
            current_phy_lambda = lambda_phy # 固定物理权重
            current_alpha = 1.0            # 固定alpha值
            current_temperature = temperature # 固定温度
            phy_desc = f", 物理权重:{current_phy_lambda:.4f}, alpha:{current_alpha:.2f}, temp:{current_temperature:.4f}"
        
        # 如果使用软对比损失，在训练阶段创建新的SoftContrastiveLoss实例
        if args.use_soft_contrast == 1:
            soft_contrast_loss = SoftContrastiveLoss(
                temperature=current_temperature,
                label_smoothing=args.label_smoothing,
                use_soft_contrast=True,
                similarity_threshold=args.similarity_threshold
            ).to(device).to(dtype=torch.bfloat16)
        else:
            soft_contrast_loss = None # 不使用软对比损失
        
        # 训练阶段
        model.train() # 设置模型为训练模式
        running_total_loss = 0.0        # 累计总损失
        running_text_recon_loss = 0.0   # 累计文本重构损失
        running_kl_loss = 0.0           # 累计KL散度损失
        running_fine_align_loss = 0.0   # 累计细粒度对齐损失
        running_coarse_align_loss = 0.0 # 累计粗粒度对齐损失
        running_phy_loss = 0.0          # 累计物理损失
        
        # 训练进度条
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                    desc=f"{stage_info}Epoch {epoch+1}/{num_epochs} [{phy_desc}]")
                    
        for i, (cst_params, texts, short_texts, physics_labels) in pbar:
            # 将数据移至指定设备并转换数据类型
            cst_params = cst_params.to(device, dtype=torch.bfloat16)
            physics_labels = physics_labels.to(device, dtype=torch.bfloat16) if physics_labels is not None else None
            
            # 模型前向传播
            text_decoded_cst, mu, logvar, text_fine, cst_fine, text_coarse, cst_coarse, physical_params = model(
                cst_params, texts, short_texts)
            
            # 计算各项损失
            total_loss, text_recon_loss, kl_loss, fine_align_loss, coarse_align_loss, phy_loss = loss_function(
                text_decoded_cst, cst_params, mu, logvar, 
                text_fine, cst_fine, 
                text_coarse, cst_coarse,
                physical_params, physics_labels, 
                lambda_recon=lambda_recon, lambda_kl=lambda_kl, 
                lambda_align=lambda_align, alpha=current_alpha, temperature=current_temperature, lambda_phy=current_phy_lambda,
                soft_contrast_loss=soft_contrast_loss
            )
            
            # 反向传播和参数更新 (由DeepSpeed处理)
            model.backward(total_loss)
            model.step()
            
            # 累加各项损失值
            running_total_loss += total_loss.item()
            running_text_recon_loss += text_recon_loss.item()
            running_kl_loss += kl_loss.item()
            running_fine_align_loss += fine_align_loss.item()
            running_coarse_align_loss += coarse_align_loss.item()
            running_phy_loss += phy_loss.item()
            
            # 获取当前学习率
            current_lr = model.optimizer.param_groups[0]['lr']
            
            # 定期打印训练信息
            if (i+1) % print_interval == 0:
                print(f"{stage_info}Step {i+1}: 学习率 = {current_lr:.6f}, 总损失 = {total_loss.item():.4f}, "
                      f"cst重构损失 = {text_recon_loss.item():.4f}, "
                      f"细粒度对齐 = {fine_align_loss.item():.4f}, "
                      f"粗粒度对齐 = {coarse_align_loss.item():.4f}, "
                      f"物理损失 = {phy_loss.item():.4f}")
            
            # 更新进度条显示信息
            pbar.set_postfix({
                "学习率": f"{current_lr:.2e}",
                "总损失": f"{total_loss.item():.4f}", 
                "cst重构": f"{text_recon_loss.item():.4f}",
                "细粒度": f"{fine_align_loss.item():.4f}",
                "粗粒度": f"{coarse_align_loss.item():.4f}",
                "物理": f"{phy_loss.item():.4f}"
            })
        
        # 计算训练集上的平均损失
        avg_total_loss = running_total_loss / len(train_dataloader)
        avg_text_recon_loss = running_text_recon_loss / len(train_dataloader)
        avg_kl_loss = running_kl_loss / len(train_dataloader)
        avg_fine_align_loss = running_fine_align_loss / len(train_dataloader)
        avg_coarse_align_loss = running_coarse_align_loss / len(train_dataloader)
        avg_phy_loss = running_phy_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval() # 设置模型为评估模式
        val_running_total_loss = 0.0        # 验证集累计总损失
        val_running_text_recon_loss = 0.0   # 验证集累计文本重构损失
        val_running_kl_loss = 0.0           # 验证集累计KL散度损失
        val_running_fine_align_loss = 0.0   # 验证集累计细粒度对齐损失
        val_running_coarse_align_loss = 0.0 # 验证集累计粗粒度对齐损失
        val_running_phy_loss = 0.0          # 验证集累计物理损失
        
        with torch.no_grad(): # 禁用梯度计算
            val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), 
                           desc=f"{stage_info}Validating Epoch {epoch+1}", leave=False) # 验证进度条
            for i, (cst_params, texts, short_texts, physics_labels) in val_pbar:
                # 将数据移至指定设备并转换数据类型
                cst_params = cst_params.to(device, dtype=torch.bfloat16)
                physics_labels = physics_labels.to(device, dtype=torch.bfloat16) if physics_labels is not None else None
                # 模型前向传播
                text_decoded_cst, mu, logvar, text_fine, cst_fine, text_coarse, cst_coarse, physical_params = model(
                    cst_params, texts, short_texts)
                
                # 计算各项损失
                total_loss, text_recon_loss, kl_loss, fine_align_loss, coarse_align_loss, phy_loss = loss_function(
                    text_decoded_cst, cst_params, mu, logvar, 
                    text_fine, cst_fine, 
                    text_coarse, cst_coarse,
                    physical_params, physics_labels, 
                    lambda_recon=lambda_recon, lambda_kl=lambda_kl, 
                    lambda_align=lambda_align, alpha=current_alpha, temperature=current_temperature, lambda_phy=current_phy_lambda,
                    soft_contrast_loss=soft_contrast_loss
                )
                
                # 累加各项损失值
                val_running_total_loss += total_loss.item()
                val_running_text_recon_loss += text_recon_loss.item()
                val_running_kl_loss += kl_loss.item()
                val_running_fine_align_loss += fine_align_loss.item()
                val_running_coarse_align_loss += coarse_align_loss.item()
                val_running_phy_loss += phy_loss.item()
                
                # 更新验证进度条显示信息
                val_pbar.set_postfix({
                    "验证损失": f"{total_loss.item():.4f}"
                })
        
        # 计算验证集上的平均损失
        avg_val_total_loss = val_running_total_loss / len(val_dataloader)
        avg_val_text_recon_loss = val_running_text_recon_loss / len(val_dataloader)
        avg_val_kl_loss = val_running_kl_loss / len(val_dataloader)
        avg_val_fine_align_loss = val_running_fine_align_loss / len(val_dataloader)
        avg_val_coarse_align_loss = val_running_coarse_align_loss / len(val_dataloader)
        avg_val_phy_loss = val_running_phy_loss / len(val_dataloader)
        
        # 打印当前轮次的训练和验证结果
        print(f"{stage_info}Epoch {epoch+1}/{num_epochs}, 训练集: 平均总损失 = {avg_total_loss:.4f}, "
              f"平均cst重构损失 = {avg_text_recon_loss:.4f}, "
              f"平均对齐损失 = {avg_fine_align_loss + avg_coarse_align_loss:.4f}, "
              f"平均物理损失 = {avg_phy_loss:.4f}")
        print(f"{stage_info}Epoch {epoch+1}/{num_epochs}, 验证集: 平均总损失 = {avg_val_total_loss:.4f}, "
              f"平均cst重构损失 = {avg_val_text_recon_loss:.4f}, "
              f"平均对齐损失 = {avg_val_fine_align_loss + avg_val_coarse_align_loss:.4f}, "
              f"平均物理损失 = {avg_val_phy_loss:.4f}")
        
        # 记录日志
        log_file_path = os.path.join(checkpoint_dir, "training_loss.log")
        epoch_log = (f"{stage_info}Epoch {epoch+1}: 训练集: 总损失 = {avg_total_loss:.4f}, cst重构损失 = {avg_text_recon_loss:.4f}, "
                    f"对齐损失 = {avg_fine_align_loss + avg_coarse_align_loss:.4f}, "
                    f"物理损失 = {avg_phy_loss:.4f}\n"
                    f"验证集: 总损失 = {avg_val_total_loss:.4f}, cst重构损失 = {avg_val_text_recon_loss:.4f}, "
                    f"对齐损失 = {avg_val_fine_align_loss + avg_val_coarse_align_loss:.4f}, "
                    f"物理损失 = {avg_val_phy_loss:.4f}")
        
        # 如果使用软对比损失，记录相似度阈值触发统计
        if args.use_soft_contrast == 1 and soft_contrast_loss is not None:
            trigger_rate = soft_contrast_loss.threshold_triggers / max(1, soft_contrast_loss.total_comparisons)
            threshold_log = (f"\n相似度阈值统计: 触发次数 = {soft_contrast_loss.threshold_triggers}, "
                           f"总比较次数 = {soft_contrast_loss.total_comparisons}, "
                           f"触发率 = {trigger_rate:.6f}")
            epoch_log += threshold_log
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(epoch_log + "\n")
        
        # 仅在主进程保存最新模型检查点
        if args.local_rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
            torch.save(model.module.state_dict(), checkpoint_path) # 保存模型参数
            print(f"{stage_info}已保存检查点: {checkpoint_path}")
            
        # 如果当前验证损失优于历史最佳损失，则保存为最佳模型 (仅在主进程)
        if args.local_rank == 0 and avg_val_total_loss < best_loss:
            best_loss = avg_val_total_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.module.state_dict(), best_checkpoint_path) # 保存模型参数
            print(f"{stage_info}已保存最佳检查点: {best_checkpoint_path}")

# 新增长短文本对比学习训练函数
def train_text_contrastive(model, train_dataloader, val_dataloader, device, num_epochs=5, 
                          print_interval=20, checkpoint_dir="./checkpoint/text_contrastive",
                          temperature=0.07, stage="stage0", args=None):
    """
    长短文本对比学习阶段的训练函数。
    此阶段主要训练文本编码器，以生成更优质的文本表示。

    参数:
        model: 训练的模型。
        train_dataloader: 训练数据加载器。
        val_dataloader: 验证数据加载器。
        device: 训练设备。
        num_epochs: 训练的总轮数。
        print_interval: 打印训练信息的间隔步数。
        checkpoint_dir: 保存模型检查点的目录。
        temperature: 对比学习的温度系数。
        stage: 当前训练阶段的标识。
        args: 命令行参数。
    """
    best_loss = float('inf') # 初始化最佳损失为正无穷大
    
    # 仅在主进程创建检查点目录
    if args.local_rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"创建文本对比学习检查点目录: {checkpoint_dir}")
    
    stage_info = f"[{stage}] " if stage else "" # 日志中阶段信息的前缀
    
    # 温度参数的初始值和最终值 (当前版本未使用动态温度)
    temp_start = 0.045
    temp_end = 0.18
    
    for epoch in range(num_epochs):
        # 为分布式训练设置采样器的epoch
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'set_epoch'):
            val_dataloader.sampler.set_epoch(epoch)
            
        # 动态计算当前温度值 (当前版本固定为0.045)
        progress_ratio = epoch / num_epochs
        #current_temperature = temp_start + (temp_end - temp_start) * min(1.0, progress_ratio * 1.5)
        current_temperature = 0.045
        # 训练阶段
        model.train() # 设置模型为训练模式
        running_loss = 0.0 # 累计损失
        
        # 训练进度条
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                    desc=f"{stage_info}Epoch {epoch+1}/{num_epochs} [temp:{current_temperature:.4f}]")
        
        for i, (_, texts, short_texts, _) in pbar:
            # 使用混合精度进行文本编码
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 获取长短文本的编码，并通过投影层
                long_text_embeddings = model.module.text_encoder.transformer.encode_text(texts, convert_to_tensor=True)
                short_text_embeddings = model.module.text_encoder.transformer.encode_text(short_texts, convert_to_tensor=True)
                
                long_embeddings = model.module.text_encoder.projection(long_text_embeddings)
                short_embeddings = model.module.text_encoder.projection(short_text_embeddings)
                
                # 计算对比损失
                loss = contrastive_loss(long_embeddings, short_embeddings, current_temperature)
            
            # 反向传播和参数更新 (由DeepSpeed处理)
            model.backward(loss)
            model.step()
            
            running_loss += loss.item() # 累加损失
            
            # 获取当前学习率
            current_lr = model.optimizer.param_groups[0]['lr']
            
            # 定期打印训练信息
            if (i+1) % print_interval == 0:
                print(f"{stage_info}Step {i+1}: 学习率 = {current_lr:.6f}, 对比损失 = {loss.item():.4f}")
            
            # 更新进度条显示信息
            pbar.set_postfix({
                "学习率": f"{current_lr:.2e}",
                "对比损失": f"{loss.item():.4f}"
            })
        
        # 计算训练集上的平均损失
        avg_loss = running_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval() # 设置模型为评估模式
        val_running_loss = 0.0 # 验证集累计损失
        
        with torch.no_grad(): # 禁用梯度计算
            val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                           desc=f"{stage_info}Validating Epoch {epoch+1}", leave=False) # 验证进度条
            
            for i, (_, texts, short_texts, _) in val_pbar:
                # 使用混合精度进行文本编码
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 获取长短文本的编码，并通过投影层
                    long_text_embeddings = model.module.text_encoder.transformer.encode_text(texts, convert_to_tensor=True)
                    short_text_embeddings = model.module.text_encoder.transformer.encode_text(short_texts, convert_to_tensor=True)
                    
                    long_embeddings = model.module.text_encoder.projection(long_text_embeddings)
                    short_embeddings = model.module.text_encoder.projection(short_text_embeddings)
                    
                    # 计算对比损失
                    loss = contrastive_loss(long_embeddings, short_embeddings, temperature)
                
                val_running_loss += loss.item() # 累加损失
                
                # 更新验证进度条显示信息
                val_pbar.set_postfix({
                    "验证损失": f"{loss.item():.4f}"
                })
        
        # 计算验证集上的平均损失
        avg_val_loss = val_running_loss / len(val_dataloader)
        
        # 打印当前轮次的训练和验证结果
        print(f"{stage_info}Epoch {epoch+1}/{num_epochs}, 训练集: 平均对比损失 = {avg_loss:.4f}")
        print(f"{stage_info}Epoch {epoch+1}/{num_epochs}, 验证集: 平均对比损失 = {avg_val_loss:.4f}")
        
        # 记录日志
        log_file_path = os.path.join(checkpoint_dir, "training_loss.log")
        epoch_log = (f"{stage_info}Epoch {epoch+1}: 训练集: 对比损失 = {avg_loss:.4f}\n"
                    f"验证集: 对比损失 = {avg_val_loss:.4f}")
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(epoch_log + "\n")
        
        # 仅在主进程保存最新模型检查点
        if args.local_rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
            torch.save(model.module.state_dict(), checkpoint_path) # 保存模型参数
            print(f"{stage_info}已保存检查点: {checkpoint_path}")
            
            # 如果当前验证损失优于历史最佳损失，则保存为最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.module.state_dict(), best_checkpoint_path) # 保存模型参数
                print(f"{stage_info}已保存最佳检查点: {best_checkpoint_path}")
    
    return model


def freeze_encoders(model):
    """
    冻结文本编码器和CST VAE编码器的参数。
    此函数用于第一阶段训练，只训练模型的投影层和解码器部分。
    """
    # 冻结文本编码器的Transformer部分参数
    for name, param in model.text_encoder.transformer.named_parameters():
        param.requires_grad = False
    print("已冻结文本编码器参数")
    
    # 冻结CST VAE编码器部分的参数 (fc1, fc2, fc_mu, fc_logvar层)
    for name, param in model.cst_vae.named_parameters():
        if name.startswith('fc1') or name.startswith('fc2') or name.startswith('fc_mu') or name.startswith('fc_logvar'):
            param.requires_grad = False
    print("已冻结CST VAE编码器参数")
    
    # 确认模型的投影层和解码器参数仍为可训练状态
    # 物理参数投影层
    for name, param in model.text_encoder.physical_projection.named_parameters():
        param.requires_grad = True
    # 文本投影层
    for name, param in model.text_encoder.projection.named_parameters():
        param.requires_grad = True
    # VAE投影层
    for name, param in model.vae_projection.named_parameters():
        param.requires_grad = True
    # 解码器 (CST VAE的解码器部分)
    for name, param in model.decoder.named_parameters():
        param.requires_grad = True
    print("投影层和解码器参数保持可训练状态")

def unfreeze_all_parameters(model):
    """解冻模型中的所有参数，使其均可训练。"""
    for name, param in model.named_parameters():
        param.requires_grad = True
    print("已解冻所有模型参数")

def main():
    # 配置参数解析器
    parser = argparse.ArgumentParser(description="CLIP图文对齐训练脚本")
    # 数据集与路径参数
    parser.add_argument("--h5_file", type=str, default="data/CLIP_dataset_cst_enhanced_PY_mix_large.h5", help="文本-CST对齐数据集h5文件路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_cst_alignment", help="检查点保存目录")
    parser.add_argument("--vae_checkpoint", type=str, default="./cstvae2_checkpoint/best_checkpoint.pth", help="预训练VAE模型路径")
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64, help="每个GPU的batch size")
    parser.add_argument("--total_epochs", type=int, default=110, help="训练的总epoch数")
    parser.add_argument("--contrastive_epochs", type=int, default=5, help="文本对比学习阶段的epoch数 (阶段0)")
    parser.add_argument("--stage1_epochs", type=int, default=10, help="第一阶段训练的epoch数 (冻结编码器)")
    # 模型结构参数
    parser.add_argument("--physics_dim", type=int, default=5, help="物理维度")
    parser.add_argument("--pretrained_model_name", type=str, default="jinaai/jina-clip-v2", help="预训练模型名称 (例如 jinaai/jina-clip-v2)")
    parser.add_argument("--truncate_dim", type=int, default=None, help="截断嵌入维度，设为None则使用完整维度")
    parser.add_argument("--latent_dim", type=int, default=512, help="CST VAE的潜空间维度")
    parser.add_argument("--shared_dim", type=int, default=512, help="文本和CST共享的潜空间维度")
    parser.add_argument("--vae_latent_dim", type=int, default=512, help="加载的预训练VAE模型的内部潜空间维度 (用于模型初始化)")
    # 损失函数权重
    parser.add_argument("--lambda_recon", type=float, default=0.5, help="重构损失权重")
    parser.add_argument("--lambda_kl", type=float, default=0.0001, help="KL散度损失权重")
    parser.add_argument("--lambda_align", type=float, default=1.0, help="对齐损失权重")
    parser.add_argument("--lambda_phy", type=float, default=0.2, help="物理维度约束损失权重")
    parser.add_argument("--alpha", type=float, default=1.0, help="粗粒度对齐损失在总对齐损失中的权重")
    parser.add_argument("--temperature", type=float, default=0.045, help="对比学习温度系数")
    # DeepSpeed 配置
    parser.add_argument("--deepspeed_config", type=str, default="config3_C.json", help="默认DeepSpeed配置文件路径 (用于全参数训练)")
    parser.add_argument("--deepspeed_config_stage1", type=str, default="config3_A.json", help="第一阶段DeepSpeed配置文件路径 (冻结编码器)")
    parser.add_argument("--deepspeed_config_stage2", type=str, default="config3_B.json", help="第二阶段DeepSpeed配置文件路径 (全参数训练，与默认可能相同或不同)")
    parser.add_argument("--local_rank", type=int, default=0, help="DeepSpeed分布式训练的local rank (通常由DeepSpeed自动设置)")
    # 其他训练设置
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保实验可复现")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], 
                       help="训练阶段选择 (0: 文本对比学习 | 1: 冻结编码器训练 | 2: 全参数训练)")
    parser.add_argument("--load_contrastive", type=int, default=0, choices=[0, 1], help="是否加载预训练的文本对比学习模型 (0: 不加载, 1: 加载)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="对比损失中的标签平滑系数")
    parser.add_argument("--similarity_threshold", type=float, default=0.45, help="软对比损失中的相似度阈值")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器使用的工作进程数")
    parser.add_argument("--use_soft_contrast", type=int, default=1, choices=[0, 1], 
                       help="是否使用软对比损失 (0: 不使用, 1: 使用)")  
    parser.add_argument("--use_variable_temperature", type=int, default=1, choices=[0, 1], 
                       help="是否在阶段2使用可变温度参数 (0: 固定温度, 1: 动态调整温度)")
    args = parser.parse_args()

    # 设置全局随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化分布式训练环境
    if "LOCAL_RANK" in os.environ: # 检查是否由DeepSpeed启动
        args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank) # 为当前进程设置CUDA设备
    dist.init_process_group(backend="nccl") # 初始化NCCL后端进行分布式通信
    device = torch.device("cuda", args.local_rank) # 获取当前进程的设备
    print(f"使用设备: {device}")

    # 加载HDF5数据集
    full_dataset = TextCSTPairH5Dataset(args.h5_file, physics_dim=args.physics_dim)
    print(f"数据集样本数: {len(full_dataset)}")
    print(f"使用物理参数维度: {args.physics_dim}")
    
    # 从数据集中获取CST参数的维度
    sample = full_dataset[0]
    cst_dim = sample['cst'].shape[0]
    print(f"CST参数维度: {cst_dim}")
    
    # 划分训练集和验证集
    train_size = int((1 - args.val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed) # 使用固定种子保证划分一致
    )
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    
    # 初始化训练集分布式采样器
    if args.local_rank != -1: # -1通常表示非分布式环境
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(), # 总GPU数量
            rank=dist.get_rank(),             # 当前GPU的rank
            shuffle=True,                     # 每个epoch打乱数据
            drop_last=False                   # 不丢弃最后一个不完整的batch
        )
    else:
        train_sampler = None # 单GPU或CPU训练

    # 创建训练数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), # 如果没有sampler (单GPU)，则进行shuffle
        num_workers=args.num_workers,    # 数据加载子进程数量
        sampler=train_sampler,           # 分布式采样器
        collate_fn=text_cst_collate_fn   # 自定义的数据整理函数
    )
    
    # 初始化验证集分布式采样器
    if args.local_rank != -1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False, # 验证集通常不打乱
            drop_last=False
        )
    else:
        val_sampler = None

    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        collate_fn=text_cst_collate_fn
    )
    
    # 初始化对齐模型
    model = TextCSTAlignmentModel(
        pretrained_model_name=args.pretrained_model_name,
        cst_dim=cst_dim,
        vae_latent_dim=args.vae_latent_dim, # CST VAE内部的潜变量维度
        shared_dim=args.shared_dim,         # 文本和CST共享的特征维度
        truncate_dim=args.truncate_dim,     # 是否截断文本/CST嵌入维度
        physics_dim=args.physics_dim,       # 物理参数维度
        pca_dim=32                          # 物理参数PCA降维后的维度
    )
    model.to(device) # 将模型移至指定设备
    
    # 加载预训练的CST VAE编码器参数
    if os.path.exists(args.vae_checkpoint):
        vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        # 从VAE检查点中提取模型状态字典
        vae_state_dict = vae_checkpoint["model_state_dict"]
        # 筛选出VAE编码器相关的参数 (通常是 fc1, fc2, fc_mu, fc_logvar 层)
        encoder_state_dict = {k: v for k, v in vae_state_dict.items() 
                             if k.startswith('fc1') or k.startswith('fc2') or 
                                k.startswith('fc_mu') or k.startswith('fc_logvar')}
        
        # 创建新的状态字典，用于加载到当前模型的cst_vae部分
        # 参数名称在源VAE模型和当前模型的CST VAE部分中应保持一致
        new_state_dict = {}
        for k, v in encoder_state_dict.items():
            new_state_dict[k] = v
        
        # 加载筛选后的编码器参数到当前模型的CST VAE中，允许部分参数不匹配 (strict=False)
        model.cst_vae.load_state_dict(new_state_dict, strict=False)
        print(f"已加载预训练VAE编码器: {args.vae_checkpoint}")
    else:
        print(f"警告: 预训练VAE模型不存在: {args.vae_checkpoint}")
        
    # 将模型所有参数强制转换为bfloat16类型以节省显存并加速训练
    model = model.to(dtype=torch.bfloat16)
    for param in model.parameters():
        param.data = param.data.to(dtype=torch.bfloat16)

    # 禁用Jina CLIP模型的FlashAttention特性 (如果存在相关配置)
    # FlashAttention可能与某些环境或DeepSpeed版本不兼容
    cfg = model.text_encoder.transformer.config # 获取文本编码器配置
    if hasattr(cfg, "text_config") and isinstance(cfg.text_config, dict):
        hf_kwargs = cfg.text_config.get("hf_model_config_kwargs", {})
        print("当前hf_kwargs:", hf_kwargs)
        hf_kwargs["use_flash_attn"] = False # 禁用FlashAttention
        cfg.text_config["hf_model_config_kwargs"] = hf_kwargs
    if hasattr(cfg, "use_text_flash_attn"):
        setattr(cfg, "use_text_flash_attn", False) # 另一种可能的禁用方式

    # 阶段0：长短文本对比学习
    if args.stage == 0:
        print("=== 开始文本对比学习阶段 ===")
        contrastive_dir = os.path.join(args.output_dir, "text_contrastive") # 该阶段检查点保存目录
        
        # 冻结模型中除文本编码器外的所有参数
        for name, param in model.named_parameters():
            param.requires_grad = False # 默认全部冻结
        for name, param in model.text_encoder.named_parameters():
            param.requires_grad = True  # 解冻文本编码器参数
             
        # 使用DeepSpeed初始化模型和优化器 (配置通常针对此阶段进行调整)
        # 注意：此时的args.deepspeed_config应指向文本对比学习阶段的配置文件
        model, optimizer, _, _ = deepspeed.initialize(args=args, model=model)
            
        # 执行文本对比学习训练
        train_text_contrastive(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            num_epochs=args.contrastive_epochs,
            checkpoint_dir=contrastive_dir,
            temperature=args.temperature, # 使用args中定义的温度
            stage="stage0",
            args=args
        )
        
        # 仅在主进程保存训练完成的文本对比模型
        if args.local_rank == 0 or args.local_rank == -1: # -1表示非分布式
            print("长短文本对比学习阶段完成")
            text_model_path = os.path.join(contrastive_dir, "text_contrastive_model.pth")
            torch.save(model.module.state_dict(), text_model_path) # 保存模型参数
            print(f"文本对比模型已保存至 {text_model_path}")
        args.stage = 1 # 自动进入下一阶段或指示下一阶段开始
    else:
        print("跳过长短文本对比学习阶段")
    
    # 第一阶段：冻结编码器参数，只训练投影层和CST VAE的解码器
    if args.stage == 1:
        args.deepspeed_config = args.deepspeed_config_stage1 #切换到第一阶段的DeepSpeed配置
        print("=== 开始冻结编码器训练 ===")
        
        # 根据参数选择是否加载在阶段0预训练的文本对比学习模型
        if args.load_contrastive == 1:
            contrastive_dir = os.path.join(args.output_dir, "text_contrastive")
            text_contrastive_model_path = os.path.join(contrastive_dir, "best_model.pth") # 加载最佳模型
            
            if os.path.exists(text_contrastive_model_path):
                print(f"发现文本对比学习模型: {text_contrastive_model_path}")
                try:
                    checkpoint = torch.load(text_contrastive_model_path, map_location=device)
                    
                    # 处理DeepSpeed保存的模型参数可能带有的 'module.' 前缀
                    if 'model_state_dict' in checkpoint: # 检查点可能是字典格式
                        checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                    else: # 检查点直接是状态字典
                        checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                        
                    # 筛选出仅与文本编码器相关的参数进行加载
                    text_encoder_dict = {k: v for k, v in checkpoint_dict.items() if k.startswith('text_encoder')}
                    
                    # 加载文本编码器参数，允许不完全匹配 (strict=False)
                    load_result = model.load_state_dict(text_encoder_dict, strict=False)
                    if load_result.missing_keys:
                        print("以下参数未加载 (可能属于模型其他部分或参数名不匹配):", load_result.missing_keys)
                    if load_result.unexpected_keys:
                        print("以下多余参数未被使用 (可能在当前模型中不存在):", load_result.unexpected_keys)
                    print(f"已成功加载文本对比学习模型")
                except Exception as e:
                    print(f"加载文本对比学习模型时出错: {e}")
            else:
                print("未发现文本对比学习模型，将使用初始模型参数进行训练")
        else:
            print("已设置不加载文本对比学习模型，将使用初始模型参数进行训练")
        
        # 冻结模型中的编码器部分参数
        freeze_encoders(model)
        print("第一阶段训练：已冻结编码器参数，只训练投影层和解码器")
        
        # 创建第一阶段的输出目录
        stage1_output_dir = os.path.join(args.output_dir, "stage1")
        if not os.path.exists(stage1_output_dir) and args.local_rank == 0: # 仅主进程创建
            os.makedirs(stage1_output_dir, exist_ok=True)
        
        # 使用DeepSpeed初始化模型和优化器 (配置针对此阶段)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args, model=model)
        
        # 执行第一阶段的训练和验证
        train_with_validation(
                model, 
                train_dataloader, 
                val_dataloader,
                device, 
                num_epochs=args.stage1_epochs, 
                checkpoint_dir=stage1_output_dir,
                lambda_recon=args.lambda_recon,
                lambda_kl=args.lambda_kl,
                lambda_align=args.lambda_align,
                lambda_phy=args.lambda_phy,
                alpha=args.alpha,
                temperature=args.temperature, # 此阶段通常使用固定的温度
                stage="stage1",
                args=args
            )
        
        # 仅在主进程保存第一阶段训练完成的模型
        if args.local_rank == 0 or args.local_rank == -1:
            stage1_model_path = os.path.join(stage1_output_dir, "stage1_model.pth")
            torch.save(model.module.state_dict(), stage1_model_path)
            print(f"第一阶段模型已保存至 {stage1_model_path}")

        # 如果总训练轮数大于第一阶段轮数，则进入第二阶段
        if args.total_epochs - args.stage1_epochs > 0:
            # 解冻模型所有参数，准备进行全参数微调
            unfreeze_all_parameters(model) # 此时model是DeepSpeed封装的对象
            print("第二阶段训练：已解冻所有参数")
            
            # 创建第二阶段的输出目录
            stage2_output_dir = os.path.join(args.output_dir, "stage2")
            if not os.path.exists(stage2_output_dir) and args.local_rank == 0: # 仅主进程创建
                os.makedirs(stage2_output_dir, exist_ok=True)
            
            # 切换到第二阶段的DeepSpeed配置并重新初始化优化器等
            # 注意：模型参数会保留，但优化器状态等会重新开始
            args.deepspeed_config = args.deepspeed_config_stage2
            model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args, model=model)
        
            # 执行第二阶段的训练和验证 (全参数微调)
            train_with_validation(
                model, 
                train_dataloader, 
                val_dataloader,
                device, 
                num_epochs=args.total_epochs - args.stage1_epochs, # 剩余的epoch数
                checkpoint_dir=stage2_output_dir,
                lambda_recon=args.lambda_recon,
                lambda_kl=args.lambda_kl,
                lambda_align=args.lambda_align,
                lambda_phy=args.lambda_phy, # 物理损失权重
                alpha=args.alpha,           # 粗细粒度对齐损失权重
                temperature=args.temperature, # 温度，可能动态调整
                stage="stage2",
                args=args
            )
            
            # 仅在主进程保存最终训练完成的模型
            if args.local_rank == 0 or args.local_rank == -1:
                final_model_path = os.path.join(stage2_output_dir, "final_model.pth")
                torch.save(model.module.state_dict(), final_model_path)
                print(f"最终模型已保存至 {final_model_path}")
    
    # 阶段2：直接开始或继续全参数训练
    # 此分支通常用于从检查点恢复第二阶段的训练，或者直接跳过前序阶段开始全参数训练
    if args.stage == 2:
        print("=== 开始全参数训练 ===")
        # 创建/指定第二阶段的输出目录
        stage2_output_dir = os.path.join(args.output_dir, "stage2")
        if not os.path.exists(stage2_output_dir) and args.local_rank == 0: # 仅主进程创建
            os.makedirs(stage2_output_dir, exist_ok=True)

        # 尝试加载第二阶段的最新检查点 (latest_model.pth)
        stage2_latest_model_path = os.path.join(stage2_output_dir, "latest_model.pth")

        model_loaded = False # 标记是否成功加载模型
        if os.path.exists(stage2_latest_model_path):
            print(f"发现第二阶段最新模型: {stage2_latest_model_path}")
            checkpoint = torch.load(stage2_latest_model_path, map_location=device)
            # 处理DeepSpeed保存的模型参数可能带有的 'module.' 前缀
            if 'model_state_dict' in checkpoint:
                checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            else:
                checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            
            load_result = model.load_state_dict(checkpoint_dict, strict=True) # 全参数加载，严格匹配
            # strict=True 会在参数名不完全匹配时报错
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print(f"已成功加载第二阶段最新模型: {stage2_latest_model_path}")
                model_loaded = True
            else:
                if load_result.missing_keys: print("加载时以下参数缺失:", load_result.missing_keys)
                if load_result.unexpected_keys: print("加载时出现以下预期外参数:", load_result.unexpected_keys)
        
        # 如果第二阶段最新模型加载失败或不存在，尝试加载第一阶段的最佳模型
        if not model_loaded:
            print(f"未找到或加载第二阶段最新模型失败，尝试加载第一阶段最佳模型...")
            stage1_checkpoint_dir = os.path.join(args.output_dir, "stage1")
            stage1_best_model_path = os.path.join(stage1_checkpoint_dir, "best_model.pth")
            
            if os.path.exists(stage1_best_model_path):
                print(f"发现第一阶段最佳模型: {stage1_best_model_path}")
                checkpoint = torch.load(stage1_best_model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                else:
                    checkpoint_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                
                load_result = model.load_state_dict(checkpoint_dict, strict=True) # 假设第一阶段模型与当前模型结构兼容
                if not load_result.missing_keys and not load_result.unexpected_keys:
                    print(f"已成功加载第一阶段最佳模型: {stage1_best_model_path}")
                    model_loaded = True
                else:
                    if load_result.missing_keys: print("加载时以下参数缺失:", load_result.missing_keys)
                    if load_result.unexpected_keys: print("加载时出现以下预期外参数:", load_result.unexpected_keys)
            else:
                print(f"未发现第一阶段最佳模型: {stage1_best_model_path}")

        if not model_loaded:
            print("警告：未成功加载任何预训练模型检查点，将从头开始训练或使用初始化的模型参数。")
        
        # 无论是否加载成功，都需要初始化DeepSpeed
        args.deepspeed_config = args.deepspeed_config_stage2 # 确保使用第二阶段的配置
        model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args, model=model)
        
        # 解冻所有参数，准备进行第二阶段（全参数）训练
        # 对于DeepSpeed封装的模型，需要操作 model.module
        unfreeze_all_parameters(model.module if hasattr(model, 'module') else model)
        print("第二阶段训练：已解冻所有参数")
        
        # 开始第二阶段（全参数）训练
        train_with_validation(
            model, 
            train_dataloader, 
            val_dataloader,
            device, 
            num_epochs=args.total_epochs, # 使用总的epoch数进行训练
            checkpoint_dir=stage2_output_dir,
            lambda_recon=args.lambda_recon,
            lambda_kl=args.lambda_kl,
            lambda_align=args.lambda_align,
            lambda_phy=args.lambda_phy,
            alpha=args.alpha,
            temperature=args.temperature,
            stage="stage2",
            args=args
        )
        
        # 仅在主进程保存最终训练完成的模型
        if args.local_rank == 0 or args.local_rank == -1:
            final_model_path = os.path.join(stage2_output_dir, "final_model.pth")
            torch.save(model.module.state_dict(), final_model_path)
            print(f"最终模型已保存至 {final_model_path}")

if __name__ == "__main__":
    main()