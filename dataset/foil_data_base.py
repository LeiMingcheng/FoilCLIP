import json
import os
import uuid
import numpy as np
from scipy.optimize import curve_fit
from cst_modeling.section import cst_foil_fit, cst_foil
import re
from cst_modeling.math import cst_curve
from enhanced_cst import enhanced_le_cst_fit, read_coordinates



def circle_equation(x, x0, y0, r):
    # 对于数值稳定性，使用 np.clip 限制平方根内部的值不小于0
    inside = np.clip(r**2 - (x - x0)**2, 0, None)
    return np.sqrt(inside) + y0

def calculate_leading_edge_radius(coordinates):
    """
    计算翼型前缘曲率半径（三点拟合法）
    前10%弦长范围内的点视为前缘区域
    """
    try:
        # 转换为numpy数组并找到最小x值点（前缘点）
        coords = np.array(coordinates)
        min_idx = np.argmin(coords[:,0])
        
        # 分离上下表面坐标（假设前一半是上表面，后一半是下表面）
        split_idx = len(coords) // 2
        upper = coords[:split_idx]    # 上表面坐标（x从0到1）
        lower = coords[split_idx:]    # 下表面坐标（x从1到0）
        num_points = 5
        # 在上表面取前3个点（靠近前缘）
        upper_points = upper[1:num_points]
        # 在下表面取最后3个点（靠近前缘）
        lower_points = lower[-num_points:]
        
        # 合并上下表面点
        points = np.vstack([upper_points, lower_points])
        
        # 三点拟合法
        x_data = points[:,0]
        y_data = points[:,1]
        
        # 动态生成初始猜测（基于前三个点）
        x0_guess = 100*np.mean(x_data[:5])  # 初始x坐标取前三个点平均值
        y0_guess = np.mean(y_data[:5])       # 初始y坐标取前三个点平均值
        r_guess = 2*np.sqrt((x_data[0]-x0_guess)**2 + (y_data[0]-y0_guess)**2)*1.5

        # 添加参数约束（圆心在前缘附近，半径合理范围）
        bounds = ((-0.1, -0.1, 0.005), (0.1, 0.1, 0.2))

        best_err = float('inf')
        best_radius = None
        factors = [0.01, 0.1, 1, 10, 100, 1000]

        for fx in factors:
            for fy in factors:
                for fr in factors:
                    initial_guess = (x0_guess * fx, y0_guess * fy, r_guess * fr)
                    try:
                        params, _ = curve_fit(circle_equation, x_data, y_data,
                                             p0=initial_guess, bounds=bounds,
                                             maxfev=5000)
                        # 计算残差平方和
                        residuals = circle_equation(x_data, params[0], params[1], params[2]) - y_data
                        err = np.sum(residuals**2)
                        if err < best_err:
                            best_err = err
                            best_radius = params[2]
                    except Exception:
                        continue
        if best_radius is not None:
            print(best_radius)
            return round(abs(best_radius), 5)
        else:
            print("所有初始猜测均失败")
            return None
    except Exception as e:
        print(f"曲率计算失败: {str(e)}")
        return None

def calculate_leading_edge_radius_poly(coordinates):
    """
    使用二次多项式拟合前缘区域的翼型数据，并计算 (0,0) 处的曲率半径。
    假设翼型的前缘点已位于 (0,0) 处。
    """
    try:
        coords = np.array(coordinates)
        split_idx = len(coords) // 2
        upper = coords[:split_idx]
        lower = coords[split_idx:]
        # 选取上表面前7个点和下表面后7个点构成前缘区域
        num_points = 20
        upper_points = upper[:num_points]
        lower_points = lower[-num_points:]
        points = np.vstack([upper_points, lower_points])
        x_data = points[:, 0]
        y_data = points[:, 1]
        # 构建设计矩阵，强制截距为 0
        A = np.vstack([x_data**2, x_data]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y_data, rcond=None)
        a, b = coeffs
        # 计算曲率: kappa = |2a| / (1 + b^2)^(3/2)
        curvature = abs(2 * a) / ((1 + b**2) ** 1.5)
        if curvature == 0:
            return None
        radius = 1 / curvature
        print(radius)
        return round(radius, 5)
    except Exception as e:
        print(f"曲率计算失败: {str(e)}")
        return None

# 新增函数：使用样条插值计算前缘曲率半径
def calculate_leading_edge_radius_spline(coordinates,num_points=15):
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
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        # 如果前缘点在上表面和下表面的连接处，需要特殊处理
        # 从上表面取点
        upper_points = coords[1:num_points]
        # 从下表面取点
        lower_points = coords[len(coords)-num_points:]
        # 合并并按x排序
        le_points = np.vstack([lower_points, upper_points])

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
            print(f"样条插值法计算的前缘曲率半径: {radius}")
            return round(abs(radius), 5)
        else:
            return None
    except Exception as e:
        print(f"样条曲率计算失败: {str(e)}")
        return None

def generate_prompt(details, polar_details):
    """
    Generate prompt based on instruction type.
    This dataset is used for Florence image-text alignment fine-tuning. 
    The airfoil images are relatively simple without complex structures,
    only requiring a brief description of the basic geometric characteristics and aerodynamic performance.
    
    Note: The final answer must strictly follow the dictionary format, including the following two keys:
      - 'question': A relevant question about the airfoil description;
      - 'answer': A concise answer about the airfoil description.
    """
    context_str = f"Brief introduction to the airfoil: {details}, Aerodynamic characteristics: {polar_details}"
    
    return (
        f"You are an airfoil expert analyzing a specific airfoil image.\n"
        f"{context_str}\n\n"
        "Based on the information above, please provide a concise description of this airfoil, focusing on its basic geometric characteristics, aerodynamic performance, and its features and application scenarios.\n"
        "Example: NASA SC(2)-0714 is a supercritical airfoil with maximum thickness of 13.9% at 37% chord, maximum camber of 2.5% at 80% chord, leading edge radius of 3.0% chord, maximum lift coefficient of 1.85, and maximum lift-to-drag ratio of 80.2. It is suitable for transonic flight, featuring good lift-drag characteristics and an upper surface design that delays shock wave formation."
        "(The sentence order can be modified)"
    )

def generate_batch_requests(data_folder):
    """
    遍历数据文件夹，生成批量请求的 JSONL 文件，添加增强型CST参数。
    """
    # 打开输出文件
    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as jsonl_file:
        # 遍历文件夹中的 JSON 文件
        for filename in os.listdir(data_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(data_folder, filename)
                print(f"Processing file: {filename}")
                
                # 读取 JSON 文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    airfoil_data = json.load(f)

                details = airfoil_data.get("Details", "")
                polar_details = airfoil_data.get("PolarDetails", [])

                # 根据json文件名构建dat文件路径
                base_name = os.path.splitext(filename)[0]
                if base_name.endswith("_data"):
                    base_name = base_name[:-5]
                dat_path = os.path.join(data_folder, f"{base_name}.dat")
                
                # CST参数字典
                cst_parameters = {}
                
                try:
                    # 读取dat文件坐标并合并上下表面
                    xu, yu, xl, yl = read_coordinates(dat_path)                               
                    
                    # 1. 增强前缘CST拟合 (基础参数+前缘修正参数)
                    cst_base_u, cst_base_l, cst_le_u, cst_le_l = enhanced_le_cst_fit(
                        xu, yu, xl, yl, n_base=10, n_diff=40, n_enhanced=3)
                    cst_parameters= {
                        "cst_base_u": cst_base_u.tolist(),
                        "cst_base_l": cst_base_l.tolist(),
                        "cst_le_u": cst_le_u.tolist(),
                        "cst_le_l": cst_le_l.tolist()
                    }

                    # 2. 标准CST拟合
                    cst_u, cst_l = cst_foil_fit(xu, yu, xl, yl, n_cst=20)
                    xx, fitted_yu, fitted_yl, _, R0 = cst_foil(101, cst_u, cst_l, x=None, t=None, tail=0.0)
                    coordinates = np.vstack([
                        np.column_stack((xx, fitted_yu)),
                        np.column_stack((xx[::-1], fitted_yl[::-1]))  # 下表面坐标反向排列
                    ]).tolist()
                    
                    # 计算前缘曲率半径
                    #radius = calculate_leading_edge_radius_spline(coordinates, num_points=3)*100
                    radius = R0*100
                    print(f"CST拟合前缘曲率半径: {R0*100:.2f}%弦长")
                    if radius is None:
                        print("样条插值法失败，尝试使用拟合圆法...")
                        radius = calculate_leading_edge_radius(coordinates)*100
                    
                    if radius:
                        details += f"，前缘曲率半径{radius:.2f}%弦长"
                    
                except Exception as e:
                    print(f"处理 {dat_path} 失败: {str(e)}")
                
                # 计算气动特性
                max_cl = 0.0
                max_ratio = 0.0
                best_alpha_cl = 0.0
                best_alpha_ratio = 0.0
                
                for polar in polar_details:
                    if "table" in polar:
                        for record in polar["table"]:
                            cl = float(record["Cl"])
                            cd = float(record["Cd"])
                            
                            # 计算最大升力系数
                            if cl > max_cl:
                                max_cl = cl
                                best_alpha_cl = float(record["Alpha"])
                            
                            # 计算升阻比（避免除以0）
                            if cd > 1e-6:
                                ratio = cl / cd
                                if ratio > max_ratio:
                                    max_ratio = ratio
                                    best_alpha_ratio = float(record["Alpha"])
                
                # 添加升力特性描述
                if max_cl > 0:
                    details += f"，最大升力系数{max_cl:.2f}（{best_alpha_cl:.0f}°攻角）"
                if max_ratio > 0:
                    details += f"，最大升阻比{max_ratio:.1f}（{best_alpha_ratio:.0f}°攻角时）"

                # 结构化存储翼型参数
                airfoil_params = {
                    "max_thickness": None,
                    "max_thickness_loc": None,
                    "max_camber": None, 
                    "max_camber_loc": None,
                    "leading_edge_radius": round(radius, 2) if radius is not None else None,
                    "max_cl": round(max_cl, 2),
                    "max_lift_ratio": round(max_ratio, 2),
                }
                
                # 提取几何参数
                if "Max thickness" in details:
                    match = re.search(r'Max thickness\s*([\d.]+)%\s*at\s*([\d.]+)%', details)
                    if match:
                        airfoil_params["max_thickness"] = float(match.group(1))
                        airfoil_params["max_thickness_loc"] = float(match.group(2))
                
                if "Max camber" in details:
                    match = re.search(r'Max camber\s*([\d.]+)%\s*at\s*([\d.]+)%', details)
                    if match:
                        airfoil_params["max_camber"] = float(match.group(1))
                        airfoil_params["max_camber_loc"] = float(match.group(2))

                # 保留特定攻角的记录
                allowed_alphas = {-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
                for polar in polar_details:
                    if "table" in polar:
                        polar["table"] = [
                            record for record in polar["table"]
                            if float(record.get("Alpha", "0")) in allowed_alphas
                        ]

                # 构造 Prompt
                prompt = generate_prompt(details, polar_details)

                # 构造请求体
                request_body = {
                    "custom_id": f"{os.path.splitext(filename)[0]}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "deepseek-r1",
                        "temperature": 0.5,
                        "airfoil_parameters": {k: v for k, v in airfoil_params.items() if v is not None},
                        "cst_parameters": cst_parameters,  # 添加CST参数
                        "messages": [
                            {"role": "system", "content": "You are an airfoil expert analyzing a specific airfoil image. Please describe from the perspective of someone who can 'see the airfoil'. Keep your response about 100 words."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                }

                # 写入 JSONL 文件
                jsonl_file.write(json.dumps(request_body, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 数据文件夹路径
    DATA_FOLDER = "../airfoil_data"
    # 输出 JSONL 文件路径
    OUTPUT_JSONL_FILE = "./json_data/airfoil_requests_base_new.jsonl"
    # 生成批量请求文件
    generate_batch_requests(DATA_FOLDER)

    print(f"批量请求文件已生成：{OUTPUT_JSONL_FILE}")