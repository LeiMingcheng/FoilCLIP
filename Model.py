#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLIP 图文对齐训练脚本（包含 CST 参数回归）

数据集需提供 npz 文件，其中包含：
  - samples: 图片数组，形状 (N, H, W, C)
  - cst: CST 参数数组（例如 20 维）
  - texts: 文本描述列表
  - airfoil_parameters: 空气动力学参数数组（例如 4 维）
  
本脚本直接使用 AutoModel.from_pretrained 加载 "jinaai/jina-clip-v2" 模型，
利用模型内部的 encode_text/encode_image 方法提供图文编码，同时在图像嵌入上加一个 CST head 回归对应的 CST 参数。
  
最终总损失 = 图文对齐损失 + lambda_cst * CST 损失 + lambda_phy * 物理约束损失。
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel
from PIL import Image
import argparse
import torch.distributed as dist
from tqdm import tqdm
deepspeed.ops.op_builder.CPUAdamBuilder().load()
import torch.distributed.nn as distnn
import h5py
import json

# 自定义数据集：加载h5文件
class TextCSTPairH5Dataset(Dataset):
    """
    自定义数据集类，用于从HDF5文件中加载文本-CST参数对。
    HDF5文件应包含以下结构:
      - 'cst/base_u': 上翼面基础CST系数
      - 'cst/base_l': 下翼面基础CST系数
      - 'cst/le_u': 上翼面前缘修正CST系数
      - 'cst/le_l': 下翼面前缘修正CST系数
      - 'text/full_text': 完整的文本描述
      - 'text/short_text': 简短的文本描述
      - 'params/airfoil': 包含翼型物理参数的JSON字符串
    HDF5文件属性中应包含 'sample_count'，表示样本总数。
    """
    def __init__(self, h5_file, physics_dim=5):
        """
        初始化数据集。

        参数:
            h5_file (str): HDF5文件的路径。
            physics_dim (int): 要加载的物理参数的维度数量。
        """
        self.h5_file = h5_file
        self.physics_dim = physics_dim
        with h5py.File(h5_file, 'r') as f:
            self.sample_count = f.attrs['sample_count'] # 获取样本总数
            
    def __len__(self):
        """返回数据集中样本的总数。"""
        return self.sample_count
    
    def __getitem__(self, idx):
        """
        根据索引获取单个样本数据。

        参数:
            idx (int): 样本的索引。

        返回:
            dict: 包含CST参数、长文本、短文本和物理参数的字典。
        """
        with h5py.File(self.h5_file, 'r') as f:
            # 获取CST参数
            base_u = f['cst/base_u'][idx]
            base_l = f['cst/base_l'][idx]
            le_u = f['cst/le_u'][idx]
            le_l = f['cst/le_l'][idx]
            
            # 将所有CST参数连接成一个向量
            cst = np.concatenate([base_u, le_u, base_l, le_l])
            
            # 获取文本数据，并确保解码为UTF-8字符串
            text_bytes = f['text/full_text'][idx]
            short_text_bytes = f['text/short_text'][idx]
            text = text_bytes.decode('utf-8') if isinstance(text_bytes, bytes) else text_bytes
            short_text = short_text_bytes.decode('utf-8') if isinstance(short_text_bytes, bytes) else short_text_bytes
            
            # 获取物理参数 (存储为JSON字符串)
            airfoil_params_json = f['params/airfoil'][idx]
            airfoil_dict = json.loads(airfoil_params_json) # 解析JSON字符串
            
            # 定义所有可用的物理参数键名 (按期望顺序)
            physics_keys = [
                "max_thickness",
                "max_thickness_loc",
                "max_camber",
                "max_camber_loc",
                "leading_edge_radius",
                # 可以根据需要继续添加更多物理参数键名
            ]
            
            # 根据指定的physics_dim提取物理参数值
            # 如果字典中不存在某个键，则使用0作为默认值
            physics_values = [
                airfoil_dict.get(key, 0) 
                for key in physics_keys[:self.physics_dim]
            ]
            physics_params = np.array(physics_values, dtype=np.float32)
            
        return {
            'cst': torch.tensor(cst, dtype=torch.float32), 
            'text': text, 
            'short_text': short_text,
            'physics_params': torch.tensor(physics_params, dtype=torch.float32)
        }


def text_cst_collate_fn(batch):
    """
    自定义的collate函数，用于将一批样本数据整理成模型输入所需的格式。

    参数:
        batch (list): 一个包含多个样本字典的列表，每个字典由TextCSTPairH5Dataset的__getitem__方法返回。

    返回:
        tuple: 包含CST参数张量、长文本列表、短文本列表和物理参数张量的元组。
               如果物理参数不存在，则物理参数张量为None。
    """
    csts = [item['cst'] for item in batch]
    texts = [item['text'] for item in batch]
    short_texts = [item['short_text'] for item in batch]
    physics = [item['physics_params'] for item in batch]
    
    # 将CST参数列表堆叠成一个张量，并转换为bfloat16类型
    csts_tensor = torch.stack(csts).to(dtype=torch.bfloat16)
    
    # 处理物理参数
    if physics[0] is not None: # 假设批次中所有样本都有或都没有物理参数
        physics_tensor = torch.tensor(np.stack(physics), dtype=torch.bfloat16)
    else:
        physics_tensor = None # 如果没有物理参数，则为None
        
    return csts_tensor, texts, short_texts, physics_tensor

class CSTVAE(nn.Module):
    """
    CST参数的变分自编码器 (VAE)。
    该VAE包含一个编码器，将输入的CST参数映射到潜空间分布 (均值和对数方差)，
    以及一个解码器，从潜空间样本重构CST参数。
    结构与预训练的VAE保持一致，以便加载预训练权重。投影层将独立定义。
    """
    def __init__(self, input_dim, latent_dim=512, hidden_dim=512, physics_dim=4):
        """
        初始化CSTVAE。

        参数:
            input_dim (int): 输入CST参数的维度。
            latent_dim (int): 潜空间的维度。
            hidden_dim (int): VAE内部隐藏层的维度。
            physics_dim (int): 物理参数的维度 (当前模型未使用，但为保持一致性保留)。
        """
        super(CSTVAE, self).__init__()
        self.physics_dim = physics_dim
        
        # 保持与预训练VAE完全相同的结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器部分（与预训练VAE保持一致）
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, physics_params=None):
        """
        编码器前向传播。

        参数:
            x (torch.Tensor): 输入的CST参数。
            physics_params (torch.Tensor, optional): 物理参数 (当前未使用)。

        返回:
            tuple: 包含潜空间均值 (mu) 和对数方差 (logvar) 的元组。
        """
        h = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧，用于从潜空间分布中采样。

        参数:
            mu (torch.Tensor): 潜空间的均值。
            logvar (torch.Tensor): 潜空间的对数方差。

        返回:
            torch.Tensor: 从潜空间采样的向量。
        """
        std = torch.exp(0.5 * logvar) # 计算标准差
        eps = torch.randn_like(std)   # 从标准正态分布中采样噪声
        return mu + eps * std         # 应用重参数化技巧
    
    def decode(self, z):
        """
        解码器前向传播。

        参数:
            z (torch.Tensor): 从潜空间采样的向量。

        返回:
            torch.Tensor: 重构的CST参数。
        """
        h = self.lrelu(self.fc_dec1(z))
        h = self.lrelu(self.fc_dec2(h))
        return self.fc_out(h)
    
    def forward(self, x, physics_params=None):
        """
        VAE的完整前向传播。

        参数:
            x (torch.Tensor): 输入的CST参数。
            physics_params (torch.Tensor, optional): 物理参数 (当前未使用)。

        返回:
            tuple: 包含重构的CST参数 (recon_x)、潜空间均值 (mu)、
                   潜空间对数方差 (logvar) 和潜空间样本 (z) 的元组。
        """
        mu, logvar = self.encode(x)           # 编码到潜空间分布
        z = self.reparameterize(mu, logvar) # 从潜空间采样
        recon_x = self.decode(z)            # 解码重构
        return recon_x, mu, logvar, z

class ProjectionLayer(nn.Module):
    """
    一个独立的投影层模块。
    用于将VAE产生的潜空间表示映射到图文共享的嵌入空间。
    """
    def __init__(self, latent_dim=512, projection_dim=512):
        """
        初始化投影层。

        参数:
            latent_dim (int): 输入潜空间的维度 (来自VAE)。
            projection_dim (int): 输出共享嵌入空间的维度。
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 1024), # 第一个全连接层，扩展维度
            nn.ReLU(),                   # ReLU激活函数
            nn.Linear(1024, projection_dim) # 第二个全连接层，映射到目标维度
        )
    
    def forward(self, z):
        """
        投影层前向传播。

        参数:
            z (torch.Tensor): 输入的潜空间表示 (来自VAE)。

        返回:
            torch.Tensor: 投影到共享嵌入空间的表示。
        """
        return self.projection(z)

class TextEncoderWithProjection(nn.Module):
    """
    文本编码器模块，集成了预训练的Transformer模型和一个投影层。
    该模块负责将输入文本编码为向量表示，然后通过投影层将其映射到
    与CST参数潜空间对齐的共享嵌入空间。同时，它还包含一个额外的
    投影层用于从文本嵌入中预测相关的物理参数。
    """
    def __init__(self, pretrained_model_name='./jina-clip2', latent_dim=512, truncate_dim=None, physics_dim=4):
        """
        初始化文本编码器与投影层。

        参数:
            pretrained_model_name (str): 预训练Transformer模型的名称或路径 (例如, "jinaai/jina-clip-v2")。
            latent_dim (int): 目标共享嵌入空间的维度。
            truncate_dim (int, optional): 如果提供，则将文本嵌入截断到此维度。默认为None。
            physics_dim (int): 要预测的物理参数的维度。
        """
        super(TextEncoderWithProjection, self).__init__()
        # 加载预训练的Transformer模型 (例如Jina CLIP)
        self.transformer = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True, # 允许加载远程代码 (如果模型需要)
        )
        self.truncate_dim = truncate_dim # 文本嵌入截断维度

        # 动态获取预训练模型的隐藏层维度 (embedding维度)
        if hasattr(self.transformer.config, "hidden_size"):
            hidden_size = self.transformer.config.hidden_size
        elif hasattr(self.transformer.config, "projection_dim"): # 某些模型使用projection_dim
            hidden_size = self.transformer.config.projection_dim
        elif hasattr(self.transformer.config, "text_config") and hasattr(self.transformer.config.text_config, "hidden_size"):
            # 某些模型 (如Jina CLIP) 将文本配置嵌套在内
            hidden_size = self.transformer.config.text_config.hidden_size
        else:
            raise AttributeError("模型配置中未找到 hidden_size 或相关属性！无法确定文本嵌入维度。")
        
        # 文本到共享潜空间的投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        # 文本到物理参数的预测层
        self.physical_projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, physics_dim)
        )
    
    def forward(self, texts):
        """
        文本编码器前向传播。

        参数:
            texts (list of str): 包含待编码文本的列表。

        返回:
            tuple: 包含投影后的文本潜向量 (latent_vectors) 和预测的物理参数 (physical_params) 的元组。
        """
        # 使用预训练模型编码文本
        # convert_to_tensor=True 表示输出直接为PyTorch张量
        text_embeddings = self.transformer.encode_text(texts, truncate_dim=self.truncate_dim, convert_to_tensor=True)
        
        # 从文本嵌入预测物理参数
        physical_params = self.physical_projection(text_embeddings)
        
        # 将文本嵌入投影到共享潜空间
        latent_vectors = self.projection(text_embeddings)
        
        return latent_vectors, physical_params

class TextCSTAlignmentModel(nn.Module):
    """
    文本-CST参数对齐的总模型。
    该模型集成了文本编码器、CST VAE、投影层、共享解码器以及PCA组件，
    旨在实现文本描述与CST几何参数之间的对齐和转换。
    采用双路径设计，分别处理细粒度和粗粒度的对齐。
    """
    def __init__(self, pretrained_model_name='./jina-clip2', cst_dim=20, 
                 vae_latent_dim=512, shared_dim=512, truncate_dim=None,
                 physics_dim=4, pca_dim=32):
        """
        初始化文本-CST对齐模型。

        参数:
            pretrained_model_name (str): 预训练文本编码器模型的名称或路径。
            cst_dim (int): CST参数的维度。
            vae_latent_dim (int): CST VAE内部潜空间的维度。
            shared_dim (int): 文本和CST共享的嵌入空间维度。
            truncate_dim (int, optional): 文本嵌入的截断维度。
            physics_dim (int): 物理参数的维度。
            pca_dim (int): PCA降维后的目标维度，用于粗粒度特征提取。
        """
        super().__init__()
        self.physics_dim = physics_dim
        self.pca_dim = pca_dim # PCA降维的目标维度
        
        # 文本编码器模块 (包含到共享空间的投影和物理参数预测)
        self.text_encoder = TextEncoderWithProjection(
            pretrained_model_name=pretrained_model_name,
            latent_dim=shared_dim,
            truncate_dim=truncate_dim,
            physics_dim=physics_dim
        )
        
        # CST VAE模块 (编码CST参数到其自身潜空间)
        self.cst_vae = CSTVAE(
            input_dim=cst_dim,
            latent_dim=vae_latent_dim, # VAE自身的潜空间维度
            hidden_dim=512,
            physics_dim=physics_dim # 保持一致性
        )
        
        # VAE潜空间到共享空间的投影层
        self.vae_projection = ProjectionLayer(
            latent_dim=vae_latent_dim, # 输入为VAE潜空间维度
            projection_dim=shared_dim  # 输出为共享空间维度
        )
        
        # 共享解码器 (从共享潜空间和物理参数解码回CST参数)
        self.decoder = CSTDecoder(
            latent_dim=shared_dim,    # 输入共享潜空间的维度
            output_dim=cst_dim,       # 输出CST参数的维度
            physics_dim=physics_dim   # 输入物理参数的维度
        )

    def PCA(self, input_tensor, PCA_dim):
        """
        对输入张量执行主成分分析 (PCA) 降维，然后再投影回原始空间。
        用于提取数据的粗粒度特征。

        参数:
            input_tensor (torch.Tensor): 待处理的输入张量 (批次大小, 特征维度)。
            PCA_dim (int): PCA降维后的目标维度。

        返回:
            torch.Tensor: 经过PCA处理后恢复到原始维度的张量。
        """
        # SVD计算通常在float32上更稳定
        input_tensor_float = input_tensor.to(torch.float32)
        
        # 数据中心化 (去均值)
        mean = torch.mean(input_tensor_float, dim=0)
        X_centered = input_tensor_float - mean.unsqueeze(0) # unsqueeze(0) 保持维度匹配

        # 使用奇异值分解 (SVD) 计算主成分
        # U: 左奇异向量, S: 奇异值, Vt: 右奇异向量的转置 (主成分)
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        principal_components = Vt.T[:, :PCA_dim] # 取前PCA_dim个主成分

        # 将数据投影到PCA选定的主成分上 (降维)
        X_transformed = torch.mm(X_centered, principal_components)
        # 将降维后的数据投影回原始特征空间 (近似恢复)
        X_reversed = torch.mm(X_transformed, principal_components.T)
        X_reversed += mean # 加回均值

        # 将结果转换回原始的bfloat16类型
        return X_reversed.to(torch.bfloat16)

    def forward(self, cst_params, texts, short_texts):
        """
        模型的前向传播，采用双路径对齐设计。

        路径1 (细粒度对齐): 使用长文本和原始CST潜空间表示。
        路径2 (粗粒度对齐): 使用短文本和经过PCA简化的CST潜空间表示。

        参数:
            cst_params (torch.Tensor): 输入的CST参数张量。
            texts (list of str): 长文本描述列表。
            short_texts (list of str): 短文本描述列表。

        返回:
            tuple: 包含以下元素的元组:
                - text_decoded_cst (torch.Tensor): 从文本潜空间解码得到的CST参数。
                - mu (torch.Tensor): CST VAE潜空间的均值。
                - logvar (torch.Tensor): CST VAE潜空间的对数方差。
                - text_fine (torch.Tensor): 长文本的细粒度嵌入表示。
                - cst_fine (torch.Tensor): CST参数的细粒度嵌入表示。
                - text_coarse (torch.Tensor): 短文本的粗粒度嵌入表示。
                - cst_coarse (torch.Tensor): CST参数的粗粒度嵌入表示 (经PCA处理)。
                - physical_params_fine (torch.Tensor): 从长文本预测的物理参数。
        """
        # 1. CST参数编码与投影
        # 通过CST VAE获取重构的CST、潜分布参数(mu, logvar)和潜空间样本(vae_z)
        recon_cst, mu, logvar, vae_z = self.cst_vae(cst_params)
        
        # 将VAE的潜空间样本(vae_z)投影到共享嵌入空间，得到细粒度CST特征(cst_fine)
        cst_fine = self.vae_projection(vae_z)
        
        # 2. 粗粒度CST特征提取
        # 对细粒度CST特征(cst_fine)应用PCA，得到粗粒度CST特征(cst_coarse)
        cst_coarse = self.PCA(cst_fine, self.pca_dim)
        
        # 3. 文本编码
        # 路径1: 长文本编码，得到细粒度文本特征(text_fine)和对应的物理参数(physical_params_fine)
        text_fine, physical_params_fine = self.text_encoder(texts)
        
        # 路径2: 短文本编码，得到粗粒度文本特征(text_coarse)和对应的物理参数(physical_params_coarse)
        # 注意：当前实现中 physical_params_coarse 未被使用，但为保持对称性而计算
        text_coarse, physical_params_coarse = self.text_encoder(short_texts)
        
        # 4. 从文本潜空间解码CST参数
        # 将细粒度的文本特征(text_fine)和从长文本预测的物理参数(physical_params_fine)拼接
        # 作为共享解码器的输入，解码得到CST参数(text_decoded_cst)
        combined_vectors_fine = torch.cat([physical_params_fine, text_fine], dim=1)
        text_decoded_cst = self.decoder(combined_vectors_fine)
        
        return (text_decoded_cst, mu, logvar, 
                text_fine, cst_fine, 
                text_coarse, cst_coarse,
                physical_params_fine) # 返回预测的物理参数以计算物理损失

class CSTDecoder(nn.Module):
    """
    CST参数解码器。
    该模块从共享潜空间的嵌入表示和预测的物理参数联合解码，重构出原始的CST参数。
    """
    def __init__(self, latent_dim, output_dim, hidden_dim=512, physics_dim=4):
        """
        初始化CST解码器。

        参数:
            latent_dim (int): 输入的共享潜空间嵌入的维度。
            output_dim (int): 输出的CST参数的维度。
            hidden_dim (int): 解码器内部隐藏层的维度。
            physics_dim (int): 输入的物理参数的维度。
        """
        super(CSTDecoder, self).__init__()
        # 第一个全连接层，输入维度为共享潜空间维度 + 物理参数维度
        self.fc1 = nn.Linear(latent_dim + physics_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 第二个全连接层
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) # LeakyReLU激活函数
        self.fc_out = nn.Linear(hidden_dim, output_dim) # 输出层，映射到CST参数维度
    
    def forward(self, z):
        """
        解码器前向传播。

        参数:
            z (torch.Tensor): 拼接后的输入张量，包含共享潜空间嵌入和物理参数。

        返回:
            torch.Tensor: 重构的CST参数。
        """
        h = self.lrelu(self.fc1(z))
        h = self.lrelu(self.fc2(h))
        return self.fc_out(h)

class SoftContrastiveLoss(nn.Module):
    """
    软对比损失函数。
    该损失函数在标准对比损失的基础上引入了对负样本的加权机制，
    根据负样本与锚点样本的相似度动态调整其在损失计算中的贡献。
    当负样本与锚点非常相似 (超过阈值) 时，其权重会降低。
    支持标签平滑和分布式训练中的嵌入向量收集。
    """
    def __init__(self, temperature=0.07, 
                 label_smoothing=0.1,
                 use_soft_contrast=False,
                 similarity_threshold=0.7,
                 k=10): # k是sigmoid函数中控制权重衰减速率的参数
        """
        初始化软对比损失。

        参数:
            temperature (float): 温度系数，用于缩放相似度得分。
            label_smoothing (float): 标签平滑系数，用于防止过拟合。
            use_soft_contrast (bool): 是否启用软对比机制 (负样本加权)。
            similarity_threshold (float): 相似度阈值。当负样本与锚点的相似度高于此阈值时，
                                         其在软对比损失中的权重会降低。
            k (float): Sigmoid函数中的缩放因子，控制权重随相似度变化的敏感度。
        """
        super().__init__()
        self.temperature = temperature # 温度系数
        self.label_smoothing = label_smoothing # 标签平滑系数
        self.use_soft_contrast = use_soft_contrast # 是否启用软对比
        self.similarity_threshold = similarity_threshold # 相似度阈值
        self.cross_entropy = nn.CrossEntropyLoss() # 标准交叉熵损失
        self.k = k # Sigmoid缩放因子
        # 用于统计相似度阈值触发情况的计数器
        self.threshold_triggers = 0 # 记录负样本相似度超过阈值的次数
        self.total_comparisons = 0  # 记录总的负样本比较次数
        
    def forward(self, embeddings_A, embeddings_B):
        """
        计算软对比损失。

        参数:
            embeddings_A (torch.Tensor): 第一组嵌入向量 (例如, 文本嵌入)。
            embeddings_B (torch.Tensor): 第二组嵌入向量 (例如, CST嵌入)。

        返回:
            torch.Tensor: 计算得到的对比损失值。
        """
        # 1. 在分布式环境中收集所有GPU上的嵌入向量
        embeddings_A_all = self._gather_embeddings(embeddings_A)
        embeddings_B_all = self._gather_embeddings(embeddings_B)
        
        # 2. L2归一化嵌入向量，使其在单位超球面上
        embeddings_A_norm = nn.functional.normalize(embeddings_A_all, dim=-1)
        embeddings_B_norm = nn.functional.normalize(embeddings_B_all, dim=-1)
        
        # 3. 计算相似度矩阵 (余弦相似度)
        # sim_matrix[i, j] 表示 embeddings_A_norm[i] 和 embeddings_B_norm[j] 的相似度
        sim_matrix = torch.matmul(embeddings_A_norm, embeddings_B_norm.t())
        
        # 4. (可选) 软对比机制：计算负样本权重
        if self.use_soft_contrast:
            # pos_mask: 对角线为True的掩码，用于标识正样本对
            pos_mask = torch.eye(len(sim_matrix), device=sim_matrix.device).bool()
            # neg_sim: 将正样本对的相似度替换为-1，以便后续处理负样本
            neg_sim = sim_matrix.masked_fill(pos_mask, -1)
            
            # 统计相似度超过阈值的负样本数量 (用于监控和分析)
            with torch.no_grad(): # 不进行梯度计算
                triggers = (neg_sim > self.similarity_threshold).sum().item()
                self.threshold_triggers += triggers
                self.total_comparisons += neg_sim.numel() # 总负样本对数量
            
            # weights: 计算每个负样本的权重。相似度越高 (越接近阈值或超过阈值)，权重越小。
            # 使用sigmoid函数平滑地调整权重。
            weights = torch.sigmoid((self.similarity_threshold - neg_sim) * self.k)
            # 归一化权重，使得每行的权重和为1 (或者可以不归一化，取决于具体策略)
            weights = weights / weights.sum(dim=1, keepdim=True) 
        
        # 5. 应用温度系数，缩放相似度得分
        sim_matrix_scaled = sim_matrix / self.temperature
        
        # 6. 准备目标标签 (用于交叉熵损失)
        # 对于InfoNCE损失，目标是使对角线元素 (正样本对) 的概率最大化
        labels = torch.arange(len(sim_matrix_scaled), device=sim_matrix_scaled.device)
        
        # 7. 计算损失
        if self.use_soft_contrast:
            # 软对比损失计算 (修改版，需要仔细推敲其数学形式的正确性)
            # logits = torch.exp(sim_matrix_scaled) # 通常用于softmax之前
            
            # 构造目标概率分布 (考虑标签平滑)
            if self.label_smoothing > 0:
                pos_targets = 1.0 - self.label_smoothing
                neg_targets = self.label_smoothing / (len(sim_matrix_scaled) - 1)
                targets = neg_targets * torch.ones_like(sim_matrix_scaled)
                # 将正样本的目标概率设置为pos_targets
                targets.scatter_(1, labels.unsqueeze(1), pos_targets)
            else:
                # 如果不使用标签平滑，目标是one-hot编码
                targets = torch.eye(len(sim_matrix_scaled), device=sim_matrix_scaled.device)
            
            # (核心软对比逻辑) 根据权重调整目标分布或损失计算
            # 此处将目标与 (1-weights) 相乘，并将加权后的相似度sigmoid值加回
            # 这一步的数学含义和效果需要仔细验证
            if self.use_soft_contrast: # 再次检查以确保逻辑清晰
                targets = targets * (1 - weights) + weights * (sim_matrix.detach().sigmoid()) # 使用原始相似度计算的权重
            
            # 计算加权后的交叉熵损失 (或类似的KL散度)
            # loss_A_to_B: 从A到B的损失，即以A为锚点，B为目标
            loss_A_to_B = -(targets * nn.functional.log_softmax(sim_matrix_scaled, dim=1)).sum(1).mean()
            # loss_B_to_A: 从B到A的损失 (对称计算)
            loss_B_to_A = -(targets.t() * nn.functional.log_softmax(sim_matrix_scaled.t(), dim=1)).sum(1).mean()
        else:
            # 标准对比损失 (InfoNCE)，支持标签平滑
            if self.label_smoothing > 0:
                # 计算平滑后的交叉熵损失
                # 第一项是与真实标签的交叉熵，第二项是与均匀分布的交叉熵 (平滑项)
                loss_A_to_B = self.cross_entropy(sim_matrix_scaled, labels) * (1 - self.label_smoothing) + \
                            self.cross_entropy(sim_matrix_scaled, torch.ones_like(sim_matrix_scaled)/len(sim_matrix_scaled)) * self.label_smoothing
                loss_B_to_A = self.cross_entropy(sim_matrix_scaled.t(), labels) * (1 - self.label_smoothing) + \
                            self.cross_entropy(sim_matrix_scaled.t(), torch.ones_like(sim_matrix_scaled)/len(sim_matrix_scaled)) * self.label_smoothing
            else:
                # 不使用标签平滑的标准交叉熵损失
                loss_A_to_B = self.cross_entropy(sim_matrix_scaled, labels)
                loss_B_to_A = self.cross_entropy(sim_matrix_scaled.t(), labels)
        
        # 总损失是两个方向损失的平均值
        return (loss_A_to_B + loss_B_to_A) / 2.0
    
    def _gather_embeddings(self, embeddings):
        """
        在分布式训练环境中，从所有GPU收集嵌入向量。
        如果不在分布式环境中，则直接返回原始嵌入。

        参数:
            embeddings (torch.Tensor): 当前GPU上的嵌入向量。

        返回:
            torch.Tensor: 从所有GPU收集并拼接后的嵌入向量。
        """    
        if dist.is_initialized(): # 检查分布式环境是否已初始化
            # 使用torch.distributed.nn.all_gather收集所有进程的embeddings
            return torch.cat(distnn.all_gather(embeddings), dim=0)
        else:
            # 非分布式环境，直接返回
            return embeddings


def contrastive_loss(embeddings_A, embeddings_B, temperature=0.07):
    """
    标准对比损失函数 (InfoNCE loss)，用于长短文本之间的对比学习。
    在分布式训练环境中，会先收集所有GPU上的嵌入向量。

    参数:
        embeddings_A (torch.Tensor): 第一组嵌入向量 (例如, 长文本嵌入)。
        embeddings_B (torch.Tensor): 第二组嵌入向量 (例如, 短文本嵌入)。
        temperature (float): 温度系数，用于缩放相似度得分。

    返回:
        torch.Tensor: 计算得到的对比损失值。
    """
    # 1. 在分布式环境中收集所有GPU上的嵌入向量
    if dist.is_initialized():
        embeddings_A_all = torch.cat(distnn.all_gather(embeddings_A), dim=0)
        embeddings_B_all = torch.cat(distnn.all_gather(embeddings_B), dim=0)
    else:
        embeddings_A_all = embeddings_A
        embeddings_B_all = embeddings_B
    
    # 2. L2归一化嵌入向量
    embeddings_A_norm = embeddings_A_all / embeddings_A_all.norm(dim=-1, keepdim=True)
    embeddings_B_norm = embeddings_B_all / embeddings_B_all.norm(dim=-1, keepdim=True)
    
    # 3. 计算相似度矩阵并应用温度系数
    # sim_matrix[i, j] 是第i个A嵌入和第j个B嵌入的相似度
    sim_matrix = torch.matmul(embeddings_A_norm, embeddings_B_norm.t()) / temperature
    
    # 4. 准备目标标签
    # 对于InfoNCE，目标是最大化对角线元素 (A[i]与B[i]匹配) 的概率
    labels = torch.arange(len(embeddings_A_all), device=embeddings_A_all.device)
    
    # 5. 计算交叉熵损失
    # loss_A_to_B: 以A为锚点，B中寻找对应正样本的损失
    loss_A_to_B = nn.CrossEntropyLoss()(sim_matrix, labels)
    # loss_B_to_A: 以B为锚点，A中寻找对应正样本的损失 (对称计算)
    loss_B_to_A = nn.CrossEntropyLoss()(sim_matrix.t(), labels)
    
    # 总损失是两个方向损失的平均值
    return (loss_A_to_B + loss_B_to_A) / 2.0

def loss_function(text_decoded_cst, cst_targets, mu, logvar, 
                 text_fine, cst_fine, 
                 text_coarse, cst_coarse,
                 physics_params, physics_labels, 
                 lambda_recon=1.0, lambda_kl=0.001, 
                 lambda_align=1.0, alpha=0.5, temperature=None, lambda_phy=0.5,
                 soft_contrast_loss=None):
    """
    计算模型的总损失，包括多个组成部分：
    1. CST重构损失: 从文本解码得到的CST参数与目标CST参数之间的均方误差。
    2. KL散度损失: CST VAE潜空间的KL散度，鼓励潜分布接近标准正态分布。
    3. 细粒度对齐损失: 长文本嵌入与原始CST嵌入之间的对比损失。
    4. 粗粒度对齐损失: 短文本嵌入与PCA处理后的CST嵌入之间的对比损失。
    5. 物理参数预测损失: 从文本预测的物理参数与真实物理参数之间的L1损失。

    参数:
        text_decoded_cst (torch.Tensor): 模型从文本解码得到的CST参数。
        cst_targets (torch.Tensor): 真实的CST参数目标。
        mu (torch.Tensor): CST VAE潜空间的均值。
        logvar (torch.Tensor): CST VAE潜空间的对数方差。
        text_fine (torch.Tensor): 长文本的细粒度嵌入。
        cst_fine (torch.Tensor): CST参数的细粒度嵌入。
        text_coarse (torch.Tensor): 短文本的粗粒度嵌入。
        cst_coarse (torch.Tensor): PCA处理后的CST参数的粗粒度嵌入。
        physics_params (torch.Tensor): 模型从文本预测的物理参数。
        physics_labels (torch.Tensor): 真实的物理参数标签。
        lambda_recon (float): CST重构损失的权重。
        lambda_kl (float): KL散度损失的权重。
        lambda_align (float): 总对齐损失的权重。
        alpha (float): 粗粒度对齐损失在总对齐损失中的相对权重 (细粒度权重为1)。
        temperature (float, optional): 用于标准对比损失的温度系数。如果提供了soft_contrast_loss，则此参数被忽略。
        lambda_phy (float): 物理参数预测损失的权重。
        soft_contrast_loss (SoftContrastiveLoss, optional): 预初始化的软对比损失模块。如果提供，则使用它计算对齐损失。

    返回:
        tuple: 包含总损失和各项单独损失的元组:
               (total_loss, text_recon_loss, kl_loss, fine_align_loss, coarse_align_loss, phy_loss)
    """
    # 1. 文本解码CST的重构损失 (均方误差 MSE)
    text_recon_loss = nn.MSELoss()(text_decoded_cst, cst_targets)
    
    # 2. KL散度损失 (衡量潜分布与标准正态分布的差异)
    #   -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. 对齐损失 (细粒度和粗粒度)
    # 根据是否提供 soft_contrast_loss 模块选择损失计算方式
    if soft_contrast_loss is not None:
        # 使用传入的软对比损失模块计算
        fine_align_loss = soft_contrast_loss(text_fine, cst_fine)
        coarse_align_loss = soft_contrast_loss(text_coarse, cst_coarse)
    else:
        # 使用标准的对比损失函数 (需要提供温度)
        if temperature is None:
            raise ValueError("当不使用 soft_contrast_loss 时，必须提供 temperature 参数。")
        fine_align_loss = contrastive_loss(text_fine, cst_fine, temperature)
        coarse_align_loss = contrastive_loss(text_coarse, cst_coarse, temperature)

    # 4. 加权总对齐损失
    # align_loss = (细粒度对齐损失) + alpha * (粗粒度对齐损失)
    align_loss = fine_align_loss + alpha * coarse_align_loss

    # 5. 物理参数对齐损失 (L1损失)
    # 仅当提供了有效的物理标签时计算
    if physics_labels is not None and not torch.isnan(physics_labels).any():
        phy_loss = nn.L1Loss()(physics_params, physics_labels)
        # 对物理损失进行截断，防止梯度爆炸或过大影响
        phy_loss = torch.clamp(phy_loss, max=100.0)
    else:
        # 如果没有物理标签或标签无效，则物理损失为0
        phy_loss = torch.tensor(0.0, device=text_fine.device, dtype=text_fine.dtype) # 保持设备和类型一致
   
    # 6. 总损失 (各项损失的加权和)
    total_loss = (lambda_recon * text_recon_loss + 
                  lambda_kl * kl_loss + 
                  lambda_align * align_loss + 
                  lambda_phy * phy_loss)

    # 7. NaN值检查与处理
    # 如果总损失计算结果为NaN (Not a Number)，则替换为一个小的可微常数值，
    # 以防止训练中断，并打印各项损失以供调试。
    if torch.isnan(total_loss):
        print(f"警告: 检测到NaN损失! "
              f"text_recon={text_recon_loss.item():.4f}, "
              f"kl={kl_loss.item():.4f}, "
              f"fine_align={fine_align_loss.item():.4f}, "
              f"coarse_align={coarse_align_loss.item():.4f}, "
              f"phy={phy_loss.item():.4f}")
        # 使用一个小的、可微分的损失值替代NaN，以允许反向传播继续
        # 确保其设备和数据类型与模型一致
        total_loss = torch.tensor(1.0, device=text_fine.device, dtype=text_fine.dtype, requires_grad=True)
    
    return total_loss, text_recon_loss, kl_loss, fine_align_loss, coarse_align_loss, phy_loss

