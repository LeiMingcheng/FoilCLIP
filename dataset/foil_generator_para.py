import os
import json
import torch
import numpy as np
import shutil
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm
import uuid
import subprocess
from scipy.interpolate import splprep, splev  # 添加样条插值所需的导入
# 导入已有模块及模型
from cst_vae import CST_VAE, latent_dim
from cst_modeling.section import cst_foil_fit, cst_foil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from enhanced_cst import enhanced_cst_foil

# 文件路径设置
INPUT_JSONL_FILE = "json_data/descriptions/foil_description_0_with_cst.jsonl"   # 原翼型及描述的 JSONL 文件
#INPUT_JSONL_FILE = "../foil_description_test.jsonl"   # 原翼型及描述的 JSONL 文件
OUTPUT_JSONL_FILE = "json_data/airfoils_requests_generated.jsonl"        # 输出包含变种翼型的 JSONL 文件
XFOIL_PATH = "XFOIL/xfoil.exe"                        # XFoil可执行文件路径，请根据实际修改
TEMP_DIR = "temp_xfoil"                                # 临时目录，用于XFoil输入输出
VAE_DIR = "vae_checkpoints/enhanced_beta0.0001_best.pth"
cst_dim = 26
# ---------------------------
# 工具函数：计算几何特性、保存 DAT 文件、调用 XFoil 分析
# ---------------------------
def calculate_airfoil_geometry(x, yu, yl):
    """
    计算翼型几何特性：最大厚度及位置，最大弯度及位置
    """
    thickness = yu - yl
    max_thickness = np.max(thickness)
    max_thickness_idx = np.argmax(thickness)
    max_thickness_pos = x[max_thickness_idx]

    camber = (yu + yl) / 2
    max_camber = np.max(np.abs(camber))
    max_camber_idx = np.argmax(np.abs(camber))
    max_camber_pos = x[max_camber_idx]

    return {
        "max_thickness": f"{max_thickness * 100:.1f}%",
        "max_thickness_pos": f"{max_thickness_pos * 100:.1f}% chord",
        "max_camber": f"{max_camber * 100:.1f}%",
        "max_camber_pos": f"{max_camber_pos * 100:.1f}% chord"
    }

def save_airfoil_dat(x, yu, yl, filename):
    """
    保存翼型坐标为XFoil可识别的.dat文件
    """
    x_upper = x[::-1][:-1]
    y_upper = yu[::-1][:-1]

    x_lower = x
    y_lower = yl

    x_combined = np.concatenate([x_upper, x_lower])
    y_combined = np.concatenate([y_upper, y_lower])

    with open(filename, 'w') as f:
        for i in range(len(x_combined)):
            f.write(f"{x_combined[i]:.6f}  {y_combined[i]:.6f}\n")

def run_xfoil_analysis(dat_file, reynolds_numbers, alphas, temp_dir):
    """
    调用 XFoil 分析翼型在不同雷诺数和攻角下的气动性能。
    采用以下策略：
      1. 先使用连续攻角计算（ASEQ 命令），如果结果有效数据达到预期（每个设定的攻角都有结果，并至少有3个有效数据），则直接返回。
      2. 如果连续计算失败或有效数据不足，则采用每个攻角单独计算的模式，针对每个攻角发送独立命令，并跳过失败的攻角。
    """
    results = []
    for Re in reynolds_numbers:
        polar_data_all = []
        output_file = os.path.join(temp_dir, f"polar_{Re}.txt")
        consecutive_success = False
        # 尝试连续攻角计算模式
        try:
            alpha_min = min(alphas)
            alpha_max = max(alphas)
            alpha_step = alphas[1] - alphas[0] if len(alphas) > 1 else 1
            if os.path.exists(output_file):
                os.remove(output_file)
            cmd_continuous = f"""\
PLOP
G

LOAD {dat_file}

OPER
ITER 200
RE {Re}
VISC {Re}
PACC
{output_file}

ASEQ {alpha_min} {alpha_max} {alpha_step}
PACC
VISC
QUIT
"""
            p = Popen([XFOIL_PATH], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            try:    
                _ = p.communicate(input=cmd_continuous.encode(), timeout=15)[0]
                #print(st)
                #print('=====================================================')

            except Exception as e:
                p.kill()
                #print(f"Re={Re} 连续攻角计算超时：{e}")
                consecutive_success = False
            # 解析连续计算结果
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 12:
                    for i in range(12, len(lines)):
                        line = lines[i].strip()
                        if line:
                            values = line.split()
                            if len(values) >= 3:
                                polar_data_all.append({
                                    "Alpha": f"{float(values[0]):.4f}",
                                    "Cl": f"{float(values[1]):.4f}",
                                    "Cd": f"{float(values[2]):.4f}",
                                    "Cdp": (f"{float(values[3]):.4f}" if len(values) > 3 else "None"),
                                    "Cm": (f"{float(values[4]):.4f}" if len(values) > 4 else "None"),
                                    "Top Xtr": (f"{float(values[5]):.4f}" if len(values) > 5 else "None"),
                                    "Btm Xtr": (f"{float(values[6]):.4f}" if len(values) > 6 else "None")
                                })
                    # 判断连续计算是否成功：要求连续模式下有效结果数量不低于设定攻角数，且至少有3个有效数据
                    if len(polar_data_all) >= (len(alphas)-5) and len(polar_data_all) >= 3:
                        consecutive_success = True
                    else:
                        consecutive_success = False
                else:
                    consecutive_success = False
            else:
                consecutive_success = False
        except Exception as e:
            #print(f"Re={Re} 连续攻角计算异常或超时：{e}")
            consecutive_success = False

        # 如果连续计算失败或有效数据不足，则逐个攻角计算
        if not consecutive_success:
            #print(f"Re={Re} 连续攻角计算有效数据不足，采用单个攻角逐个计算模式")
            polar_data_all = []  # 使用单独攻角计算，清空之前的数据
            for alpha in alphas:
                individual_output = os.path.join(temp_dir, f"polar_{Re}_{alpha}.txt")
                if os.path.exists(individual_output):
                    os.remove(individual_output)
                cmd_individual = f"""\
PLOP
G

LOAD {dat_file}

OPER
ITER 200
RE {Re}
VISC {Re}
PACC
{individual_output}

ALFA {alpha}
PACC
VISC
QUIT
"""
                p = Popen([XFOIL_PATH], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                try:    
                    st = p.communicate(input=cmd_individual.encode(), timeout=2)[0]
                    #print(st)
                except Exception as e:
                    p.kill()
                    #print(f"Re={Re} 的 XFoil 在攻角 {alpha}° 单独计算时超时或异常：{e}")
                    continue
                finally:
                    # 确保进程被终止
                    try:
                        if p.poll() is None:
                            p.terminate()
                            p.wait(timeout=1)
                    except:
                        try:
                            p.kill()
                        except:
                            pass

                try:
                    with open(individual_output, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 12:
                        for i in range(12, len(lines)):
                            line = lines[i].strip()
                            if line:
                                values = line.split()
                                if len(values) >= 3:
                                    polar_data_all.append({
                                        "Alpha": f"{float(values[0]):.4f}",
                                        "Cl": f"{float(values[1]):.4f}",
                                        "Cd": f"{float(values[2]):.4f}",
                                        "Cdp": (f"{float(values[3]):.4f}" if len(values) > 3 else "None"),
                                        "Cm": (f"{float(values[4]):.4f}" if len(values) > 4 else "None"),
                                        "Top Xtr": (f"{float(values[5]):.4f}" if len(values) > 5 else "None"),
                                        "Btm Xtr": (f"{float(values[6]):.4f}" if len(values) > 6 else "None")
                                    })
                except Exception as e:
                    #print(f"Re={Re} 在攻角 {alpha}° 单独计算结果读取失败：{e}")
                    continue

                # 清理单个攻角的临时文件
                try:
                    if os.path.exists(individual_output):
                        os.remove(individual_output)
                except:
                    pass

        # 筛选仅保留指定攻角的数据
        filtered = [entry for entry in polar_data_all if float(entry["Alpha"]) in alphas]
        results.append({"Reynolds": Re, "table": filtered})
        
        # 清理当前雷诺数的临时文件
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
    return results

# ---------------------------
# 生成提示文本，包含原翼型描述、当前变种翼型的几何与气动数据
# ---------------------------
def calculate_leading_edge_radius_spline(coordinates,num_points=25):
    """
    使用样条插值方法计算翼型前缘曲率半径，这是航空领域更常用的方法
    """
    from scipy.interpolate import splprep, splev
    try:
        # 转换为numpy数组
        coords = np.array(coordinates)
        split_idx = len(coords) // 2
        
        # 找到前缘点（x最小的点）
        min_x_idx = np.argmin(coords[:, 0])
        # 提取前缘附近的点（前缘前后各取10个点）
        start_idx = max(0, min_x_idx - num_points)
        end_idx = min(min_x_idx + num_points, len(coords) - 1)
        # 如果前缘点在上表面和下表面的连接处，需要特殊处理
        # 从上表面取点
        upper_points = coords[1:num_points]
        # 从下表面取点
        lower_points = coords[len(coords)-num_points:]
        # 合并并按x排序
        le_points = np.vstack([lower_points, upper_points])

        #print(f"le_points: {le_points}")
        # 确保有足够的点进行样条插值
        if len(le_points) < 4:
            return None
            
        # 使用样条插值，参数s控制平滑度
        tck, u = splprep([le_points[:, 0], le_points[:, 1]], s=0.0001, k=3)
        
        # 生成更密集的点用于计算曲率
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)
        
        # 计算一阶和二阶导数
        dx_dt, dy_dt = splev(u_fine, tck, der=1)
        d2x_dt2, d2y_dt2 = splev(u_fine, tck, der=2)
        
        # 参数化曲线曲率公式: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**(1.5)
        
        # 找到前缘点（x最小的点）
        le_point_idx = np.argmin(x_fine)
        
        # 在前缘点处的曲率
        le_curvature = curvature[le_point_idx]
        
        # 曲率半径 = 1/曲率
        radius = 1.0 / le_curvature if le_curvature > 0 else None
        
        if radius is not None:
            #print(f"样条插值法计算的前缘曲率半径: {radius}")
            return round(abs(radius), 5)
        else:
            return None
    except Exception as e:
        print(f"样条曲率计算失败: {str(e)}")
        return None

def generate_prompt_with_original(original_description, geo_info, polar_details, airfoil_params):
    """
    Construct a prompt text with:
    1. Input original airfoil description (e.g., original airfoil name, basic geometric parameters)
    2. Input current variant airfoil's geometric information and XFoil aerodynamic analysis results
    Finally emphasize: This variant airfoil is derived from the original airfoil, please provide a concise description (within 100 words)
    """
    # Add descriptions for leading edge radius, max lift coefficient, and max lift-to-drag ratio
    additional_info = ""
    if airfoil_params.get("leading_edge_radius") is not None:
        additional_info += f", leading edge radius {airfoil_params['leading_edge_radius']:.2f}% chord"
    if airfoil_params.get("max_cl") is not None:
        additional_info += f", maximum lift coefficient {airfoil_params['max_cl']:.2f} (at {airfoil_params['best_alpha_cl']:.0f}° angle of attack)"
    if airfoil_params.get("max_lift_ratio") is not None:
        additional_info += f", maximum lift-to-drag ratio {airfoil_params['max_lift_ratio']:.1f} (at {airfoil_params['best_alpha_ratio']:.0f}° angle of attack)"
    
    context_str = (
        f"Original airfoil description: {original_description} \n"
        f"Variant airfoil geometric information: {geo_info}{additional_info} \n"
        f"Aerodynamic characteristics: {polar_details} (Re=1e6 test data) "
    )
    prompt = (
        "You are an airfoil expert analyzing a specific airfoil image. "
        "Based on the following information, briefly describe the airfoil characteristics in under 100 words, focusing on its basic geometric features, aerodynamic performance, and application scenarios. Avoid lengthy analysis.\n"
        "Note that this airfoil is a *variant* derived from the original airfoil (not necessarily optimized), so please reference the original airfoil description.\n"
        f"{context_str}\n\n"
        "Only mention the original airfoil name, *series*, and application scenarios. *No need to compare with the original airfoil*, but analyze its characteristics independently. Avoid lengthy analysis."
        "Output example: This airfoil is derived from NASA SC(2)-0714, a transonic supercritical airfoil with maximum thickness of 13.9% at 37% chord, maximum camber of 2.5% at 80% chord, leading edge radius of 3.0% chord, maximum lift coefficient of 1.85, and maximum lift-to-drag ratio of 80.2. It features good lift-to-drag characteristics and upper surface design that delays shock wave formation, suitable for transonic passenger or transport aircraft cruising."
    )
    return prompt

def generate_prompt_for_npy(geo_info, polar_details, airfoil_params):
    """
    Custom prompt template designed specifically for npy base airfoils
    """
    context_str = (
        f"Geometric characteristics: {geo_info}.\n"
        f"Aerodynamic properties: {polar_details} (Re=1e6 test data).\n"
        f"Leading edge radius: {airfoil_params['leading_edge_radius']:.2f}% chord, "
        f"maximum lift coefficient {airfoil_params['max_cl']:.2f} (at {airfoil_params['best_alpha_cl']:.0f}°), "
        f"maximum lift-to-drag ratio {airfoil_params['max_lift_ratio']:.1f}"
    )
    
    prompt = (
        "You are analyzing a parametrically generated baseline airfoil. Please provide a professional description based on the following characteristics:\n"
        "1. Summarize the airfoil type in one sentence (e.g., high-lift airfoil, laminar airfoil, etc.)\n"
        "2. Detail the geometric features (thickness distribution, camber distribution, leading edge shape)\n" 
        "3. Analyze aerodynamic characteristics (lift curve, stall characteristics, lift-to-drag ratio)\n"
        "4. Infer suitable applications (UAVs, helicopter rotors, etc.)\n"
        "Requirements: Use accurate technical terminology, analyze the correlation between features and performance, avoid comparisons with other airfoils, limit to 150 words."
        f"\n\n{context_str}"
    )
    return prompt

# ---------------------------
# 主流程：读取 JSONL、生成变种、几何、气动并输出新 JSONL
# ---------------------------
def process_single_airfoil(record, vae, device, n_variants=5, noise_std=0.1):
    """
    处理单个翼型并生成其变种
    
    参数:
    - record: 原始翼型记录
    - vae: VAE模型
    - device: 计算设备
    - n_variants: 要生成的变种数量
    - noise_std: 噪声标准差
    
    返回:
    - 生成的变种翼型列表
    """
    # 创建进程独立的临时目录
    process_id = os.getpid()
    temp_dir = f"./temp_xfoil/temp_{process_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    reynolds_numbers = [1e6]
    alphas = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    # 从原记录中提取原翼型描述与CST参数
    cst_data = record.get("response", {}).get("body", {}).get("cst_parameters", {})
    # 直接读取enhanced_cst格式的参数
    base_u = np.array(cst_data.get("cst_base_u", []))
    base_l = np.array(cst_data.get("cst_base_l", []))
    le_u = np.array(cst_data.get("cst_le_u", []))
    le_l = np.array(cst_data.get("cst_le_l", []))
    print(f"base_u: {base_u}")
    print(f"base_l: {base_l}")
    print(f"le_u: {le_u}")
    print(f"le_l: {le_l}")
    # 检查参数是否合法
    if len(base_u) == 0 or len(base_l) == 0 or len(le_u) == 0 or len(le_l) == 0:
        print(f"翼型 {record.get('custom_id', '未知')} 参数格式不是enhanced_cst格式，跳过。")
        return []

    # 组合成总的参数向量
    original_params = np.concatenate([base_u, le_u, base_l, le_l], axis=0).astype(np.float32)

    # 获取参数维度（用于后续拆分）
    n_base_u = len(base_u)
    n_le_u = len(le_u)
    n_base_l = len(base_l)
    n_le_l = len(le_l)
    
    # 转为 tensor，并增加 batch 维度
    sample_tensor = torch.from_numpy(original_params).unsqueeze(0).to(device)
    # 由 VAE 编码器获取潜变量均值 mu
    with torch.no_grad():
        encoded = vae.encoder(sample_tensor)
        mu = vae.fc_mu(encoded)  # shape: [1, latent_dim]
    
    # 原翼型的描述文本：从 response 中提取
    original_description = ""
    response = record.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices", [])
    if choices:
        # 取第一条 choice 中的 message 内容
        message = choices[0].get("message", {})
        original_description = message.get("content", "")
    if not original_description:
        original_description = "暂无描述"
    
    # 针对当前翼型生成 n_variants 个有效变种翼型
    valid_variants = []
    valid_variant_count = 0
    attempt_count = 0
    max_attempts = n_variants * 2  
    consecutive_skips = 0

    # 添加单个翼型的进度条
    with tqdm(total=n_variants, desc=f"生成 {record.get('custom_id','未知')} 变种", leave=False) as pbar:
        while valid_variant_count < n_variants and attempt_count < max_attempts:
            attempt_count += 1
            noise = torch.randn_like(mu) * noise_std
            z_variant = mu + noise
            with torch.no_grad():
                decoder_in = vae.decoder_input(z_variant)
                variant_params = vae.decoder(decoder_in)  # 得到的为 [1, 20]
            variant_params = variant_params.squeeze(0).cpu().numpy()
            # 分离上下表面参数和基础/前缘参数
            cst_base_u = variant_params[:n_base_u]
            cst_le_u = variant_params[n_base_u:n_base_u+n_le_u]
            cst_base_l = variant_params[n_base_u+n_le_u:n_base_u+n_le_u+n_base_l]
            cst_le_l = variant_params[n_base_u+n_le_u+n_base_l:]
            
            # 生成翼型坐标 - 修改调用方式适应新的参数结构
            x, yu, yl, _, _ = enhanced_cst_foil(101, cst_base_u, cst_base_l, cst_le_u, cst_le_l, x=None, t=None, tail=0.0)
            # 部分处理：确保同一 x 对应 yu >= yl
            swap_mask = yu < yl
            yu[swap_mask], yl[swap_mask] = yl[swap_mask], yu[swap_mask]
            
            # 计算几何特性
            geometry = calculate_airfoil_geometry(x, yu, yl)
            cst_u, cst_l = cst_foil_fit(x, yu, x, yl, n_cst=20)
            #xx, fitted_yu, fitted_yl, _, R0 = cst_foil(101, cst_u, cst_l, x=None, t=None, tail=0.0)
            # 计算前缘曲率半径
            coordinates = np.vstack([
                np.column_stack((x, yu)),
                np.column_stack((x[::-1], yl[::-1]))
            ]).tolist()
            radius = calculate_leading_edge_radius_spline(coordinates, num_points=5)
            #radius = R0
            if radius is not None:
                radius = radius * 100  # 转换为百分比
                if radius > 20:
                    radius = 20
                if radius < 0.1:
                    radius = 0.1
            
            # 保存DAT文件供XFoil使用
            variant_id = str(uuid.uuid4())
            dat_file = os.path.join(temp_dir, f"{variant_id}.dat").replace("\\", "/")
            #print(dat_file)
            save_airfoil_dat(x, yu, yl, dat_file)
            
            # 调用XFoil进行气动分析
            polar_results = run_xfoil_analysis(dat_file, reynolds_numbers, alphas, temp_dir)
            if not polar_results:
                consecutive_skips += 1
                if consecutive_skips >= 5:
                    print(f"\n基础翼型 {record.get('custom_id', '未知')} 连续{consecutive_skips}次变种被跳过")
                    break
                continue

            # 对获得的极坐标数据进行后处理
            total_valid_angles = 0
            for polar in polar_results:
                valid_table = []
                for entry in polar["table"]:
                    valid = True
                    for key, value in entry.items():
                        if key == "Alpha" or value == "None":
                            continue
                        try:
                            num = float(value)
                            if abs(num) > 1000:
                                valid = False
                                break
                        except Exception:
                            continue
                    if valid:
                        valid_table.append(entry)
                polar["table"] = valid_table
                total_valid_angles += len(valid_table)
            
            if total_valid_angles < 3:
                consecutive_skips += 1
                if consecutive_skips >= 5:
                    print(f"\n基础翼型 {record.get('custom_id', '未知')} 连续{consecutive_skips}次变种被跳过")
                    break
                continue

            # 有效变种计数增加时更新进度条
            valid_variant_count += 1
            pbar.update(1)
            consecutive_skips = 0

            # 计算最大升力系数和最大升阻比
            max_cl = 0.0
            max_ratio = 0.0
            best_alpha_cl = 0.0
            best_alpha_ratio = 0.0
            
            for polar in polar_results:
                if "table" in polar:
                    for entry in polar["table"]:
                        try:
                            cl = float(entry["Cl"])
                            cd = float(entry["Cd"])
                            alpha = float(entry["Alpha"])
                            
                            # 计算最大升力系数
                            if cl > max_cl:
                                max_cl = cl
                                best_alpha_cl = alpha
                            
                            # 计算升阻比（避免除以0）
                            if cd > 1e-6:
                                ratio = cl / cd
                                if ratio > max_ratio:
                                    max_ratio = ratio
                                    best_alpha_ratio = alpha
                        except (ValueError, KeyError):
                            continue
            
            # 创建翼型参数字典
            airfoil_params = {
                "max_thickness": float(geometry['max_thickness'].strip('%')),
                "max_thickness_loc": float(geometry['max_thickness_pos'].split('%')[0]),
                "max_camber": float(geometry['max_camber'].strip('%')),
                "max_camber_loc": float(geometry['max_camber_pos'].split('%')[0]),
                "leading_edge_radius": round(radius, 2) if radius is not None else None,
                "max_cl": round(max_cl, 2) if max_cl > 0 else None,
                "max_lift_ratio": round(max_ratio, 1) if max_ratio > 0 else None,
                "best_alpha_cl": best_alpha_cl,
                "best_alpha_ratio": best_alpha_ratio
            }

            # 对获得的气动数据简单格式化
            polar_str = json.dumps(polar_results, ensure_ascii=False)
            # 构造几何描述字符串
            geo_str = f"最大厚度 {geometry['max_thickness']} 在 {geometry['max_thickness_pos']}，最大弯度 {geometry['max_camber']} 在 {geometry['max_camber_pos']}"
            
            # 生成提示文本
            prompt_text = generate_prompt_with_original(original_description, geo_str, polar_str, airfoil_params)
            
            # 构造变种翼型记录
            variant_record = {
                "custom_id": variant_id,
                "parent_id": record.get("custom_id", "未知"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "deepseek-r1",
                    "temperature": 0.5,
                    "airfoil_parameters": {k: v for k, v in airfoil_params.items() if v is not None},
                    "cst_parameters": {
                        "cst_base_u": cst_base_u.tolist(),
                        "cst_base_l": cst_base_l.tolist(),
                        "cst_le_u": cst_le_u.tolist(),
                        "cst_le_l": cst_le_l.tolist()
                    },
                    "messages": [
                        {"role": "system", "content": "你是一个翼型专家，正在分析一个翼型，请你以仿佛*看得见*的视角，简要描述这个翼型，重点强调几何与气动特性（例如低弯度、高升力等），最终描述不超过100字。"},
                        {"role": "user", "content": prompt_text}
                    ]
                }
            }
            
            valid_variants.append(variant_record)
    
    # 清理本进程的临时目录
    shutil.rmtree(temp_dir)
    
    if valid_variant_count < n_variants:
        print(f"翼型 {record.get('custom_id', '未知')} 生成有效变种不足{n_variants}个，仅生成 {valid_variant_count} 个。")
    
    return valid_variants

def main(n_variants=5, noise_std=0.1, consume_mode=True, num_workers=None):
    """
    主函数：并行处理多个翼型
    
    参数:
    - n_variants: 每个原始翼型生成的变种数量
    - noise_std: 噪声标准差
    - consume_mode: 是否续写模式
    - num_workers: 并行工作进程数，默认为None（使用CPU核心数）
    """
    # 如果未指定工作进程数，使用CPU核心数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"使用 {num_workers} 个CPU核心并行处理")
    
    # 创建临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 加载VAE模型
    device = torch.device("cpu")
    print(f"使用设备：{device}")
    
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    checkpoint = torch.load(VAE_DIR, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)
    vae.eval()
    
    # 备份原始 JSONL 文件
    backup_file = OUTPUT_JSONL_FILE + ".bak"
    if os.path.exists(OUTPUT_JSONL_FILE) and not os.path.exists(backup_file):
        shutil.copy(OUTPUT_JSONL_FILE, backup_file)
        print(f"已备份原始文件到 {backup_file}")
    
    # 读取输入 JSONL 文件
    input_lines = []
    if os.path.exists(INPUT_JSONL_FILE):
        with open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    input_lines.append(json.loads(line))
    
    # 自动续写逻辑
    start_index = 0
    if consume_mode:
        if os.path.exists(OUTPUT_JSONL_FILE):
            # 读取最后一行获取parent_id
            last_parent_id = None
            with open(OUTPUT_JSONL_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        last_parent_id = record.get("parent_id")
                    except:
                        continue
            # 在输入文件中定位parent_id的位置
            if last_parent_id:
                for idx, rec in enumerate(input_lines):
                    if rec.get("custom_id") == last_parent_id:
                        start_index = idx + 1  # 从下一个开始
                        print(f"检测到续写模式，从第 {start_index} 个基础翼型继续")
                        break
    
    # 准备要处理的翼型列表
    airfoils_to_process = input_lines[start_index:]
    print(f"准备处理 {len(airfoils_to_process)} 个基础翼型")
    # 以追加模式打开输出文件
    with open(OUTPUT_JSONL_FILE, "a" if start_index > 0 else "w", encoding="utf-8") as fout:
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_airfoil = {
                executor.submit(process_single_airfoil, record, vae, device, n_variants, noise_std): record
                for record in airfoils_to_process
            }
            
            # 使用tqdm显示总体进度
            with tqdm(total=len(airfoils_to_process), desc="处理基础翼型") as main_pbar:
                for future in as_completed(future_to_airfoil):
                    main_pbar.update(1)
                    record = future_to_airfoil[future]
                    try:
                        variants = future.result()
                        # 将结果写入输出文件
                        for variant in variants:
                            fout.write(json.dumps(variant, ensure_ascii=False) + "\n")
                            fout.flush()
                    except Exception as e:
                        print(f"处理翼型时发生错误: {str(e)}")
                        continue
    
    # 清理临时目录
    shutil.rmtree(TEMP_DIR)
    print(f"共生成变种翼型保存在 {OUTPUT_JSONL_FILE}")



def validate_and_supplement_variants():
    """
    后处理：检查每个基础翼型生成的变种个数
    1. 如果变种数量小于50个，尝试补充到50个
    2. 如果某个基础翼型在output中完全不存在变种，则尝试生成50个变种
    """
    print("开始验证和补充变种翼型数量...")
    
    # 加载原始翼型数据
    input_airfoils = {}
    if os.path.exists(INPUT_JSONL_FILE):
        with open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    input_airfoils[record.get("custom_id", "")] = record
    
    print(f"输入文件中共有 {len(input_airfoils)} 个基础翼型")
    
    # 统计每个基础翼型的变种数量
    variants_count = {}
    generated_variants = {}
    
    if os.path.exists(OUTPUT_JSONL_FILE):
        with open(OUTPUT_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    parent_id = record.get("parent_id", "")
                    if parent_id:
                        if parent_id not in variants_count:
                            variants_count[parent_id] = 0
                            generated_variants[parent_id] = []
                        variants_count[parent_id] += 1
                        generated_variants[parent_id].append(record)
    
    print(f"输出文件中包含 {len(variants_count)} 个基础翼型的变种数据")
    
    # 找出完全没有变种的基础翼型
    missing_parents = []
    for parent_id in input_airfoils.keys():
        if parent_id not in variants_count:
            missing_parents.append(parent_id)
    
    print(f"发现 {len(missing_parents)} 个基础翼型没有任何变种生成")
    
    # 加载VAE模型
    device = torch.device("cpu")
    print(f"使用设备：{device}")
    
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    checkpoint = torch.load(VAE_DIR, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)
    vae.eval()
    
    # 补充生成变种不足的翼型
    with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as fout:
        # 先处理完全缺失的基础翼型
        for parent_id in missing_parents:
            print(f"翼型 {parent_id} 没有任何变种，尝试生成50个变种")
            try:
                variants = process_single_airfoil(
                    input_airfoils[parent_id], 
                    vae, 
                    device, 
                    n_variants=50,
                    noise_std=0.15  # 对于完全没有变种的翼型，使用更大的噪声
                )
                
                # 写入新生成的变种
                for variant in variants:
                    fout.write(json.dumps(variant, ensure_ascii=False) + "\n")
                    fout.flush()
                
                print(f"已为翼型 {parent_id} 成功生成 {len(variants)} 个变种")
                
            except Exception as e:
                print(f"为翼型 {parent_id} 生成变种时出错: {str(e)}")
        
        # 再处理变种数量不足的翼型
        for parent_id, count in sorted(variants_count.items(), key=lambda x: x[1]):
            if count < 50 and parent_id in input_airfoils:
                additional_needed = 50 - count
                print(f"翼型 {parent_id} 仅有 {count} 个变种，需要补充 {additional_needed} 个")
                
                # 基于当前噪声增加一点噪声
                noise_std = 0.15   # 变种不足时增加噪声
                
                # 尝试生成额外的变种
                try:
                    additional_variants = process_single_airfoil(
                        input_airfoils[parent_id], 
                        vae, 
                        device, 
                        n_variants=additional_needed,
                        noise_std=noise_std
                    )
                    
                    # 写入新生成的变种
                    for variant in additional_variants:
                        fout.write(json.dumps(variant, ensure_ascii=False) + "\n")
                        fout.flush()
                    
                    print(f"已为翼型 {parent_id} 成功补充 {len(additional_variants)} 个变种")
                    
                except Exception as e:
                    print(f"为翼型 {parent_id} 补充变种时出错: {str(e)}")
    
    print("验证和补充过程完成")


if __name__ == "__main__":
    # 设置并行工作进程数，可以根据需要调整
    num_workers = multiprocessing.cpu_count() - 2  # 使用所有可用CPU核心
    #print(f"使用 {num_workers} 个CPU核心并行处理")
    # 或者指定固定数量
    #num_workers = 2  # 使用4个CPU核心
    
    # 调用主函数，传入并行工作进程数
    #main(n_variants=100, noise_std=0.15, num_workers=num_workers, consume_mode=False)
    #process_npy(num_workers=num_workers)
    validate_and_supplement_variants()