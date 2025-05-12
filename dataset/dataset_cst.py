import numpy as np
import os
from tqdm import tqdm
from cst_modeling.section import cst_foil_fit
from cst_modeling.math import cst_curve
from enhanced_cst import enhanced_le_cst_fit, read_coordinates, fit_curve_custom


def generate_combined_dataset_cst(foil_file="../CLIP/data/foil.npy", 
                                  coord_folder="../airfoil_data/dat", 
                                  output_file="dataset/airfoil_cst.npz", 
                                  n_base=7,
                                  n_diff=20,
                                  use_enhanced=True,
                                  xn1=0.5,
                                  xn2=1.0,
                                  n_enhanced=3 ):
    """
    合并读取两个来源的翼型数据，并生成含CST参数的数据集。
    
    数据来源:
        1. 训练/验证数据：从foil_file中加载翼型数据。
        2. 测试数据：从coord_folder中读取翼型坐标文件。
    
    如果use_enhanced为True，则使用增强的CST拟合方法：
        - 返回基础翼型CST参数(n_base维)和前缘修型参数(共3维)
    否则使用标准CST拟合方法：
        - 返回标准CST参数(n_base维)
    
    参数:
        foil_file: 包含翼型数据的.npy文件路径
        coord_folder: 包含翼型坐标文件的文件夹路径
        output_file: 保存生成合并数据集的文件名
        n_base: 基础CST参数阶数
        n_diff: 差异拟合CST参数阶数(仅增强模式使用)
        use_enhanced: 是否使用增强的CST拟合方法
        xn1, xn2: CST形状参数
    """
    if use_enhanced:
        labels_base_u = []
        labels_base_l = []
        labels_le_u = []
        labels_le_l = []
    else:
        labels = []
    
    # 第一部分：处理foil.npy中的翼型数据
    try:
        foil = np.load(foil_file)
        N, M = foil.shape[0], foil.shape[1]
        total_samples = N * M
        print(f"从 {foil_file} 中处理 {total_samples} 个翼型。")
        # 从N中随机采样索引
        sampled_i = np.random.choice(N, 20, replace=False)
        # 遍历采样结果
        for i in tqdm(sampled_i, desc="处理foil.npy中的第i组数据"):
            sampled_j = np.random.choice(M, 1, replace=False)
            for j in sampled_j:
                sample = foil[i, j, :, :]
                xu0 = sample[:, 0]
                yu0 = sample[:, 1]
                xl0 = sample[:, 0]
                yl0 = sample[:, 2]
                try:
                    if use_enhanced:
                        cst_base_u, cst_base_l, cst_le_u, cst_le_l = enhanced_le_cst_fit(
                            xu0, yu0, xl0, yl0, n_base=n_base, n_diff=n_diff, xn1=xn1, xn2=xn2, n_enhanced=n_enhanced)
                        
                        # 检查参数范围
                        if (np.any(cst_base_u > 5) or np.any(cst_base_l > 5) or 
                            np.any(cst_le_u > 5) or np.any(cst_le_l > 5)):
                            print(f"foil.npy样本(i={i}, j={j})中存在大于5的CST参数")
                            continue
                            
                        labels_base_u.append(cst_base_u)
                        labels_base_l.append(cst_base_l)
                        labels_le_u.append(cst_le_u)
                        labels_le_l.append(cst_le_l)
                    else:
                        cst_u, cst_l = cst_foil_fit(xu0, yu0, xl0, yl0, n_cst=n_base, xn1=xn1, xn2=xn2)
                        label = np.concatenate([cst_u, cst_l])
                        if np.any(label > 5):
                            print(f"foil.npy样本(i={i}, j={j})中存在大于5的CST参数")
                            continue
                        labels.append(label)
                except Exception as e:
                    print(f"foil.npy样本(i={i}, j={j})处理失败: {e}")
    except Exception as e:
        print(f"加载{foil_file}失败: {e}")
    
    # 第二部分：处理坐标文件夹中的翼型数据
    try:
        files = sorted([os.path.join(coord_folder, f) for f in os.listdir(coord_folder)
                        if f.endswith(".txt") or f.endswith(".dat")])
        print(f"从{coord_folder}中处理{len(files)}个翼型文件。")
        for file in tqdm(files, desc="处理测试文件"):
            try:
                xu0, yu0, xl0, yl0 = read_coordinates(file)
                if use_enhanced:
                    cst_base_u, cst_base_l, cst_le_u, cst_le_l = enhanced_le_cst_fit(
                        xu0, yu0, xl0, yl0, n_base=n_base, n_diff=n_diff, xn1=xn1, xn2=xn2)
                    
                    # 检查参数范围
                    if (np.any(cst_base_u > 5) or np.any(cst_base_l > 5) or 
                        np.any(cst_le_u > 5) or np.any(cst_le_l > 5)):
                        print(f"文件{file}中存在大于5的CST参数")
                        continue
                        
                    labels_base_u.append(cst_base_u)
                    labels_base_l.append(cst_base_l)
                    labels_le_u.append(cst_le_u)
                    labels_le_l.append(cst_le_l)
                else:
                    cst_u, cst_l = cst_foil_fit(xu0, yu0, xl0, yl0, n_cst=n_base, xn1=xn1, xn2=xn2)
                    label = np.concatenate([cst_u, cst_l])
                    if np.any(label > 5):
                        print(f"文件{file}中存在大于5的CST参数")
                        continue
                    labels.append(label)
            except Exception as e:
                print(f"文件{file}处理失败: {e}")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
    
    # 保存数据
    try:
        if use_enhanced:
            # 将所有标签转换为numpy数组
            labels_base_u = np.array(labels_base_u)
            labels_base_l = np.array(labels_base_l)
            labels_le_u = np.array(labels_le_u)
            labels_le_l = np.array(labels_le_l)
            
            # 保存增强CST参数
            np.savez(output_file, 
                    base_u=labels_base_u, 
                    base_l=labels_base_l, 
                    le_u=labels_le_u, 
                    le_l=labels_le_l)
            
            total_samples = len(labels_base_u)
            print(f"增强CST数据已保存至'{output_file}'，共包含{total_samples}个样本。")
            print(f"参数维度: 基础上表面({labels_base_u.shape[1]}), 基础下表面({labels_base_l.shape[1]}), "
                  f"前缘修型上表面({labels_le_u.shape[1]}), 前缘修型下表面({labels_le_l.shape[1]})")
        else:
            # 保存标准CST参数
            labels = np.array(labels)
            np.savez(output_file, labels=labels)
            print(f"标准CST数据已保存至'{output_file}'，共包含{labels.shape[0]}个样本。")
            print(f"参数维度: {labels.shape[1]}")
    except Exception as e:
        print(f"保存{output_file}失败: {e}")

if __name__ == "__main__":
    # 使用增强的CST拟合方法生成数据集
    generate_combined_dataset_cst(
        output_file="dataset/airfoil_enhanced_cst.npz", 
        n_base=10,
        n_diff=40,
        use_enhanced=True,
        xn1=0.5,
        xn2=1.0
    )
        