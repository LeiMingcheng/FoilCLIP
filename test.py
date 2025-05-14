"""
文本-CST对齐模型测试脚本：
1. 文本到CST参数生成：输入文本描述，生成对应的CST参数
2. CST参数分类：输入CST参数，从候选文本中找出最匹配的描述
3. 文本相似度评估：计算不同文本描述在共享潜空间的相似度

请根据需要调整检查点路径、候选文本等超参数。
"""

import numpy as np
import torch
import argparse
from PIL import Image
import os
from tqdm import tqdm

# 从项目核心模型文件导入模型类
from Model import TextCSTAlignmentModel #, CSTVAE, TextEncoderWithProjection, CSTDecoder # 保留其他可能的导入，如果它们也来自Model.py

def text_to_cst(model, input_text):
    """
    文本到CST参数生成：
    1. 将输入文本通过文本编码器编码到共享潜空间
    2. 通过解码器将潜空间表示转换为CST参数
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # 获取文本编码和物理参数
        text_latent, physical_params = model.text_encoder([input_text])
        
        # 将文本编码和物理参数拼接起来作为解码器的输入
        combined_input = torch.cat([physical_params, text_latent], dim=1)
        
        # 通过解码器生成CST参数
        cst_parameters = model.decoder(combined_input)
        # 转换为numpy数组并使用array2string格式化输出
        cst_array = cst_parameters.to(torch.float32).cpu().numpy()[0]
        cst_str = np.array2string(cst_array, separator=', ')
        return cst_str, physical_params.to(torch.float32).cpu().numpy()[0]

def classify_cst(model, cst_vector, candidate_texts, temperature=0.07):
    """
    CST参数分类：
    1. 将CST参数通过VAE编码到潜空间，再通过投影层映射到共享潜空间
    2. 计算与候选文本编码的相似度，返回最匹配的文本
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # 将CST参数转为张量并添加batch维度
        cst_tensor = torch.tensor(cst_vector, dtype=torch.bfloat16).unsqueeze(0).to(device)
        
        # 通过VAE编码获取潜空间表示
        _, _, _, vae_z = model.cst_vae(cst_tensor)
        
        # 通过投影层映射到共享潜空间
        cst_shared_latent = model.vae_projection(vae_z)
        
        # 归一化
        cst_norm = cst_shared_latent / cst_shared_latent.norm(dim=-1, keepdim=True)
        
        # 编码候选文本
        text_embeddings, physical_params = model.text_encoder(candidate_texts)
        text_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        logits = torch.matmul(cst_norm, text_norm.t()) / temperature
        
        # 打印所有候选文本的相似度分数
        print("\n各候选文本相似度分数:")
        for i, (text, score) in enumerate(zip(candidate_texts, logits[0].to(torch.float32).cpu().numpy())):
            print(f"{i+1}. 相似度: {score:.4f} - {text}")
        
        # 获取最佳匹配
        best_idx = logits[0].argmax().item()
        scores = logits[0].to(torch.float32).cpu().numpy()
        best_candidate = candidate_texts[best_idx]
        best_physical_params = physical_params[best_idx].to(torch.float32).cpu().numpy()
        
        return best_candidate, scores, best_physical_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints_cst_alignment_pre_mixd5/stage2/latest_model.pth", help="模型检查点路径")
    parser.add_argument("--local_rank", type=int, default=0, help="DeepSpeed分布式训练的local rank")
    parser.add_argument("--pretrained_model_name", type=str, default="./jina-clip2", help="预训练模型名称或路径")
    parser.add_argument("--cst_dim", type=int, default=20, help="CST参数维度")
    parser.add_argument("--vae_latent_dim", type=int, default=512, help="VAE内部潜空间维度")
    parser.add_argument("--shared_dim", type=int, default=512, help="共享潜空间维度")
    parser.add_argument("--physics_dim", type=int, default=5, help="物理参数维度")
    parser.add_argument("--truncate_dim", type=int, default=None, help="截断嵌入维度")
    parser.add_argument("--candidates", type=str, default="低雷诺数滑翔机翼型/跨音速运输机翼型/超临界翼型/水翼/NLF自然层流翼型", help="待分类候选文本，使用'/'分隔")
    parser.add_argument("--test_cst_path", type=str, default=None, help="测试CST参数文件路径(.npy)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = TextCSTAlignmentModel(
        pretrained_model_name=args.pretrained_model_name,
        cst_dim=args.cst_dim,
        vae_latent_dim=args.vae_latent_dim,
        shared_dim=args.shared_dim,
        truncate_dim=args.truncate_dim,
        physics_dim=args.physics_dim
    )
    model.to(device)
    
    # 显式设置bfloat16精度
    model = model.to(dtype=torch.bfloat16)
    
    # 加载检查点（修复分布式训练参数名）
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    print(f"成功加载检查点: {args.checkpoint_path}")
    
    # 准备测试数据
    candidate_list = [s.strip() for s in args.candidates.split("/")]
    
    # 测试文本到CST参数生成
    test_texts =  test_texts = [
        "Supercritical airfoil",
        "Supercritical airfoil, flat upper surface and aft loading camber design",
        "Whitcomb supercritical designed airfoil, max thickness 11% at 35% chord, aft camber 2.4% at 82.5% chord, relatively flat upper surface with aft loading camber design to delay shockwave generation, suitable for high subsonic/transonic cruise, typically used in main wing sections of modern mainline aircraft.",
        "The Whitcomb Integral Supercritical airfoil features a maximum thickness of 11% at 35% chord and 2.4% camber peaking near the trailing edge (82.5% chord), with a leading-edge radius of 2.44% chord. Aerodynamically, it achieves a maximum lift coefficient of 1.76 at 16° AoA and a peak lift-to-drag ratio of 77.0 at 8° AoA (Re=1×10⁶). Its flattened upper surface delays shock formation in transonic flow, while rear-loaded camber enhances lift efficiency. However, drag rises sharply beyond 8° AoA due to early flow separation on the upper surface. Designed for transonic applications, this airfoil optimizes cruise efficiency in high-subsonic aircraft, balancing reduced wave drag with stable lift performance. Typical uses include advanced transport wings and research configurations.",
        "Supercritical airfoil, max thickness 15% at 40% chord, aft camber 2.5% at 85% chord, ensuring high lift-to-drag ratio, flat upper surface, aft loading camber design, delays shockwave generation",
        "Airfoil suitable for transonic transport aircraft",
        "Airfoil suitable for transonic transport aircraft, max thickness 14% at 40% chord, aft loading camber 2% at 60% chord, increases lift",
        "Airfoil suitable for transonic transport aircraft, max thickness 11.1% at 40% chord, aft loading camber 1.4% at 60% chord, appropriately increases aft loading camber design, increases lift, delays shockwave",
        "High-efficiency low-speed glider airfoil, especially increases lift",
        "glider airfoil, especially reduces drag",
        "High-efficiency Low-speed Airfoil suitable for DLGs and gliders, max thickness 7% at 25% chord, high lift-to-drag ratio",
        "Glider airfoil based on Drela design, max thickness 5.5% at 25% chord, max camber 2% at 30% chord.",
        "Natural laminar flow airfoil based on NASA NLF series, extends laminar flow region",
        "Natural laminar flow airfoil based on NASA NLF series, reduces drag while maintaining high lift",
        "Natural laminar flow airfoil based on NASA NLF series, max thickness 15% at 39.8% chord, aft camber 4.3% concentrated at 62.8% chord, leading edge radius 1.3% chord. Its blunt leading edge design and mid-chord thickness distribution effectively extend the laminar flow region",
        "Natural laminar flow airfoil based on NASA NLF series, max thickness 15% at 44.1% chord, max camber 1.8% at 30% chord, reduces drag, extends laminar flow region",
        "Natural laminar flow airfoil based on NASA NLF series, max thickness 10% at 45% chord, relatively low max camber located in the mid-section, blunt leading edge design, gentle pressure gradient on the forward section of the airfoil, extends laminar flow region",
        "flying wing airfoil",
        "Airfoil designed for flying wing layout, featuring reflex characteristics, but ensuring a high lift coefficient.",
        "Laminar flow airfoil designed for flying wing layout, with maximum thickness of 12% and maximum camber of 3.5% both located at 30% chord.",
        "Aerobatic airfoil",
        "Aerobatic airfoil, symmetrical configuration, high lift-to-drag ratio, stable at mid-range angles of attack, sharp leading edge for high maneuverability.",
        "Aerobatic airfoil, symmetrical configuration, high lift-to-drag ratio, stable at mid-range angles of attack, high thickness for structural strength.",
        "Wind turbine airfoil, high lift-to-drag ratio.",
        "Althaus airfoil for use on wind turbines, with max thickness 20% and max camber 4%.",
        "Symmetrical thin wings with a thickness of less than 6% suitable for high-speed fighters"
    ]
    
    print("\n========== 文本到CST参数生成测试 ==========")
    for text in test_texts:
        print(f"\n输入文本: {text}")
        cst_params, physical_params = text_to_cst(model, text)
        print(f"生成的CST参数: {cst_params}")
        print(f"预测的物理参数: 最大厚度={physical_params[0]:.4f}, 最大厚度位置={physical_params[1]:.4f}, "
              f"最大弯度={physical_params[2]:.4f}, 最大弯度位置={physical_params[3]:.4f}")
    
    # 使用示例CST参数
    print("\n========== CST参数分类测试（使用示例参数）==========")
    example_cst = np.array( [[ 0.21777344,  0.09912109,  0.296875  ,  0.01422119,  0.41992188,
  0.01525879,  0.37695312,  0.18945312,  0.31640625,  0.3359375 ,
 -0.01916504,  0.02636719, -0.04296875, -0.2109375 , -0.10058594,
 -0.28320312, -0.04589844, -0.23925781, -0.18164062, -0.11279297,
 -0.04370117,  0.10595703,  0.32226562, -0.03369141,  0.07470703,
 -0.07763672],
[ 0.15332031,  0.17871094,  0.19238281,  0.140625  ,  0.2734375 ,
  0.14746094,  0.20507812,  0.29492188,  0.08447266,  0.18945312,
  0.00701904,  0.00148773,  0.00817871, -0.13671875, -0.08007812,
 -0.28320312,  0.26171875, -0.8828125 ,  0.82421875, -1.1328125 ,
  0.5546875 , -0.46484375,  0.09570312,  0.00162506,  0.00939941,
  0.00860596]])
    for cst in example_cst:
        best_match, scores, physical_params = classify_cst(model, cst, candidate_list)
        print(f"\n最佳匹配文本: {best_match}")
        print(f"预测的物理参数: 最大厚度={physical_params[0]:.4f}, 最大厚度位置={physical_params[1]:.4f}, "
          f"最大弯度={physical_params[2]:.4f}, 最大弯度位置={physical_params[3]:.4f}")


if __name__ == "__main__":
    main()