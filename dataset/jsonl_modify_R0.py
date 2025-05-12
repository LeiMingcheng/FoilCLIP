import json
import numpy as np
import os
from scipy.interpolate import splprep, splev
from tqdm import tqdm
from enhanced_cst import enhanced_cst_foil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from enhanced_cst import enhanced_le_cst_fit, read_coordinates
from cst_modeling.section import cst_foil_fit, cst_foil
# 文件路径设置
INPUT_OUTPUT_JSONL_FILE = "json_data/airfoils_requests_generated_en_modified.jsonl"
BACKUP_FILE = "json_data/airfoils_requests_generated_en_modified.jsonl.bak"

def calculate_leading_edge_radius_spline(coordinates, num_points=25):
    """
    使用样条插值方法计算翼型前缘曲率半径，这是航空领域更常用的方法
    """
    try:
        # 转换为numpy数组
        coords = np.array(coordinates)
        
        # 找到前缘点（x最小的点）
        min_x_idx = np.argmin(coords[:, 0])
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
            return round(abs(radius), 5)
        else:
            return None
    except Exception as e:
        print(f"样条曲率计算失败: {str(e)}")
        return None

def process_single_record(record):
    """处理单个翼型记录，重新计算前缘半径"""
    try:
        # 获取CST参数
        cst_params = record.get("body", {}).get("cst_parameters", {})
        if not cst_params:
            return record, False
            
        cst_base_u = np.array(cst_params.get("cst_base_u", []))
        cst_base_l = np.array(cst_params.get("cst_base_l", []))
        cst_le_u = np.array(cst_params.get("cst_le_u", []))
        cst_le_l = np.array(cst_params.get("cst_le_l", []))
        
        # 检查参数是否合法
        if len(cst_base_u) == 0 or len(cst_base_l) == 0 or len(cst_le_u) == 0 or len(cst_le_l) == 0:
            return record, False
        
        # 生成翼型坐标
        x, yu, yl, _, _ = enhanced_cst_foil(101, cst_base_u, cst_base_l, cst_le_u, cst_le_l, x=None, t=None, tail=0.0)
        
        # 处理：确保同一 x 对应 yu >= yl
        swap_mask = yu < yl
        yu[swap_mask], yl[swap_mask] = yl[swap_mask], yu[swap_mask]
        cst_u, cst_l = cst_foil_fit(x, yu, x, yl, n_cst=20)
        xx, fitted_yu, fitted_yl, _, R0 = cst_foil(101, cst_u, cst_l, x=None, t=None, tail=0.0)
        # 计算前缘曲率半径
        coordinates = np.vstack([
            np.column_stack((x, yu)),
            np.column_stack((x[::-1], yl[::-1]))
        ]).tolist()
        
        #radius = calculate_leading_edge_radius_spline(coordinates, num_points=5)
        radius = R0
        if radius is not None:
            radius = radius * 100  # 转换为百分比
            if radius > 20:
                radius = 20
            if radius < 0.1:
                radius = 0.1
            
            # 更新记录中的前缘半径
            old_radius = None
            if "airfoil_parameters" in record.get("body", {}):
                old_radius = record["body"]["airfoil_parameters"].get("leading_edge_radius")
                record["body"]["airfoil_parameters"]["leading_edge_radius"] = round(radius, 2)
                return record, True, old_radius, round(radius, 2)
        
        return record, False
        
    except Exception as e:
        print(f"处理记录 {record.get('custom_id', '未知ID')} 时出错: {str(e)}")
        return record, False

def update_leading_edge_radius(num_workers=None):
    """使用多进程并行更新JSONL文件中所有翼型的前缘半径"""
    
    # 设置并行进程数
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 2)  # 留出一个核心给系统
    
    print(f"使用 {num_workers} 个CPU核心并行处理")
    
    # 首先备份原始文件
    if not os.path.exists(BACKUP_FILE):
        import shutil
        shutil.copy(INPUT_OUTPUT_JSONL_FILE, BACKUP_FILE)
        print(f"已备份原始文件到 {BACKUP_FILE}")
    
    # 读取JSONL文件的所有记录
    records = []
    with open(INPUT_OUTPUT_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"共读取 {len(records)} 条记录")
    
    # 初始化更新计数和样本记录
    updated_count = 0
    sample_updates = []
    
    # 使用ProcessPoolExecutor进行并行处理
    updated_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有处理任务
        futures = {executor.submit(process_single_record, record): i for i, record in enumerate(records)}
        
        # 使用tqdm显示总体进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="更新前缘半径"):
            record_idx = futures[future]
            try:
                result = future.result()
                if len(result) >= 3:  # 有完整的更新结果返回
                    record, updated, old_radius, new_radius = result
                    updated_records.append(record)
                    
                    if updated:
                        updated_count += 1
                        # 收集前10个更新样本
                        if len(sample_updates) < 10:
                            sample_updates.append((record.get('custom_id'), old_radius, new_radius))
                else:
                    record, updated = result
                    updated_records.append(record)
                    if updated:
                        updated_count += 1
            except Exception as e:
                print(f"处理第 {record_idx} 条记录时出错: {str(e)}")
                updated_records.append(records[record_idx])  # 保留原记录
    
    # 确保记录顺序保持不变
    updated_records = [record for _, record in sorted(zip([futures[f] for f in futures], updated_records))]
    
    # 显示前几个样本更新
    for custom_id, old_radius, new_radius in sample_updates:
        print(f"记录 {custom_id}: 前缘半径从 {old_radius} 更新为 {new_radius}")
    
    # 写入更新后的记录
    with open(INPUT_OUTPUT_JSONL_FILE, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"已更新 {updated_count} 条记录的前缘半径")

import re
import json
from tqdm import tqdm

def update_content_leading_edge_radius(jsonl_file, output_file=None):
    """
    专门更新JSONL文件中消息内容中的前缘半径描述，保持airfoil_parameters中的值不变
    
    参数:
    - jsonl_file: 输入JSONL文件路径
    - output_file: 输出JSONL文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = jsonl_file
        
    # 备份原始文件
    backup_file = jsonl_file + ".content_bak"
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy(jsonl_file, backup_file)
        print(f"已备份原始文件到 {backup_file}")
    
    # 读取JSONL文件的所有记录
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"共读取 {len(records)} 条记录")
    
    # 定义匹配前缘半径的正则表达式模式
    patterns = [
        # 英文模式: "leading edge radius X.XX% chord"
        r'leading edge radius (\d+\.\d+)% chord',
        
        # 中文模式: "前缘半径 X.XX% 弦长" 或 "前缘半径X.XX%弦长"
        r'前缘半径 (\d+\.\d+)% 弦长',
        r'前缘半径(\d+\.\d+)%弦长',
        
        # 简化的中文模式: "前缘半径 X.XX%" 或 "前缘半径X.XX%"
        r'前缘半径 (\d+\.\d+)%',
        r'前缘半径(\d+\.\d+)%',
        
        # 其他可能的中文变体
        r'前缘曲率半径 (\d+\.\d+)%',
        r'前缘曲率半径为 (\d+\.\d+)%'
    ]
    
    # 计数器
    updated_count = 0
    sample_updates = []
    
    # 处理每条记录
    for record in tqdm(records, desc="更新前缘半径描述"):
        # 获取airfoil_parameters中的前缘半径值
        if "body" in record and "airfoil_parameters" in record["body"]:
            params = record["body"]["airfoil_parameters"]
            if "leading_edge_radius" in params:
                new_radius = params["leading_edge_radius"]
                
                # 更新body.messages中的内容
                if "messages" in record["body"]:
                    for message in record["body"]["messages"]:
                        if "content" in message:
                            content = message["content"]
                            old_content = content
                            
                            # 使用各种模式匹配并替换
                            for pattern in patterns:
                                match = re.search(pattern, content)
                                if match:
                                    old_radius = match.group(1)
                                    # 替换找到的半径值
                                    content = re.sub(
                                        pattern, 
                                        lambda m: m.group(0).replace(m.group(1), f"{new_radius:.2f}"), 
                                        content
                                    )
                                    message["content"] = content
                                    
                                    # 如果是第一次找到并更新，保存示例
                                    if old_content != content and len(sample_updates) < 10:
                                        sample_updates.append((
                                            record.get('custom_id', 'unknown'),
                                            old_radius,
                                            f"{new_radius:.2f}"
                                        ))
                                        updated_count += 1
                                        break
                
                # 更新body.choices中的消息内容
                if "choices" in record.get("body", {}):
                    for choice in record["body"]["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            
                            # 使用各种模式匹配并替换
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    choice["message"]["content"] = re.sub(
                                        pattern, 
                                        lambda m: m.group(0).replace(m.group(1), f"{new_radius:.2f}"), 
                                        content
                                    )
                
                # 更新response.body.choices中的消息内容
                if "response" in record and "body" in record["response"] and "choices" in record["response"]["body"]:
                    for choice in record["response"]["body"]["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            
                            # 使用各种模式匹配并替换
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    choice["message"]["content"] = re.sub(
                                        pattern, 
                                        lambda m: m.group(0).replace(m.group(1), f"{new_radius:.2f}"), 
                                        content
                                    )
    
    # 显示示例更新
    if sample_updates:
        print("\n前缘半径描述更新示例:")
        for custom_id, old_radius, new_radius in sample_updates:
            print(f"记录 {custom_id}: 前缘半径描述从 {old_radius}% 更新为 {new_radius}%")
    
    # 写入更新后的记录
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"已更新 {updated_count} 条记录的前缘半径描述")
    return updated_count

if __name__ == "__main__":
    import os
    # 替换为你的JSONL文件路径
    update_content_leading_edge_radius(INPUT_OUTPUT_JSONL_FILE)
    #update_leading_edge_radius()