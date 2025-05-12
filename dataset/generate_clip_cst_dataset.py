import os
import json
import numpy as np
import h5py
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import random
from pprint import pprint
import datetime

def load_json_records(jsonl_path):
    """
    读取 JSONL 文件，返回一个字典，键为 custom_id，值为完整的 JSON 记录。
    每一行均为一个有效的 JSON 对象。
    """
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            json_obj = json.loads(line)
            custom_id = json_obj.get("custom_id")
            if custom_id:
                records[custom_id] = json_obj
    return records

def parse_cst_parameters(record):
    """
    从完整的 JSON 记录中解析增强型 CST 参数，返回基础CST和前缘修型参数。
    
    返回:
    --------
    cst_base_u, cst_base_l: ndarray
        基础翼型的上下表面CST参数
    cst_le_u, cst_le_l: ndarray
        前缘修型的上下表面CST参数
    """
    body = record["response"]["body"]
    cst_params = body["cst_parameters"]
    
    # 检查是否为增强型CST格式
    if "cst_base_u" in cst_params and "cst_base_l" in cst_params and "cst_le_u" in cst_params and "cst_le_l" in cst_params:
        # 增强型CST格式
        cst_base_u = cst_params["cst_base_u"]
        cst_base_l = cst_params["cst_base_l"]
        cst_le_u = cst_params["cst_le_u"]
        cst_le_l = cst_params["cst_le_l"]
        if len(cst_base_u) != 10 or len(cst_base_l) != 10:
            raise ValueError("基础CST参数列表的长度应为10")
        if len(cst_le_u) != 3 or len(cst_le_l) != 3:
            raise ValueError("前缘修型CST参数列表的长度应为3")
        return cst_base_u, cst_base_l, cst_le_u, cst_le_l
    else:
        raise ValueError("未找到有效的CST参数结构")

def process_record(item):
    """
    处理单个记录的函数，用于并行处理。
    新增支持增强型CST参数结构。
    """
    custom_id, record = item
    try:
        cst_base_u, cst_base_l, cst_le_u, cst_le_l = parse_cst_parameters(record)
        
        # 直接拼接列表，不转换为numpy数组
        cst_vector = cst_base_u + cst_le_u + cst_base_l + cst_le_l
        
        # 参数有效性检查
        if any(abs(x) > 20 for x in cst_vector):
            print(f"过滤 {custom_id}: CST参数超出阈值")
            return None
    except Exception as e:
        print(f"记录 {custom_id}: CST参数解析错误 - {str(e)}")
        return None

    # 获取文本内容 - 修正JSON路径
    try:
        # 正确的路径：response.body.choices[0].message.content
        choices = record["response"]["body"]["choices"]
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            text = message.get("content", "")
            if not text or not text.strip():
                print(f"记录 {custom_id}: content字段为空")
                return None
        else:
            print(f"记录 {custom_id}: choices列表为空")
            return None
    except Exception as e:
        print(f"记录 {custom_id}: text字段获取错误 - {str(e)}")
        print(f"JSON结构: {record.get('response', {}).get('body', {})}")
        return None
    
    # 新增短文本描述提取
    try:
        short_desc = record["response"]["body"]["short_text"]
        if not short_desc or not short_desc.strip():
            print(f"记录 {custom_id}: short_text字段为空")
    except KeyError as e:
        print(f"记录 {custom_id}: short_text字段不存在")
        short_desc = ""

    # 提取airfoil参数
    try:
        airfoil_params = record["response"]["body"]["airfoil_parameters"]
        if not airfoil_params:
            print(f"记录 {custom_id}: airfoil_parameters字段为空字典")
    except KeyError as e:
        print(f"记录 {custom_id}: airfoil_parameters字段不存在")
        airfoil_params = {}

    return {
        "text": text,
        "short_description": short_desc,
        "cst_base_u": cst_base_u,
        "cst_base_l": cst_base_l,
        "cst_le_u": cst_le_u,
        "cst_le_l": cst_le_l,
        "airfoil_params": airfoil_params
    }

def main():
    jsonl_path = "./json_data/descriptions/foil_description_full.jsonl"
    output_h5 = "./CLIP_dataset_cst_enhanced_PY_mix_large.h5"
    
    records = load_json_records(jsonl_path)
    
    texts_all = []
    short_descriptions_all = []
    cst_base_u_all = []
    cst_base_l_all = []
    cst_le_u_all = []
    cst_le_l_all = []
    airfoil_params_all = []
    
    num_workers = 10
    print(f"使用 {num_workers} 个CPU核心并行处理")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_record = {executor.submit(process_record, item): item[0] for item in records.items()}
        for future in tqdm(concurrent.futures.as_completed(future_to_record), total=len(records), desc="处理记录"):
            custom_id = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    # 检查所有字段有效性并打印具体哪个字段为空
                    text_valid = len(result["text"].strip()) > 0
                    short_desc_valid = len(result["short_description"].strip()) > 0
                    cst_valid = (len(result["cst_base_u"]) == 10 and 
                                 len(result["cst_base_l"]) == 10 and 
                                 len(result["cst_le_u"]) == 3 and 
                                 len(result["cst_le_l"]) == 3)
                    airfoil_valid = len(result["airfoil_params"]) > 0
                    
                    if not text_valid:
                        print(f"记录 {custom_id}: text字段为空")
                    if not short_desc_valid:
                        print(f"记录 {custom_id}: short_text字段为空")
                    if not cst_valid:
                        print(f"记录 {custom_id}: CST参数无效")
                    if not airfoil_valid:
                        print(f"记录 {custom_id}: airfoil_params字段为空")
                    
                    if text_valid and cst_valid and airfoil_valid and short_desc_valid:
                        texts_all.append(result["text"])
                        short_descriptions_all.append(result["short_description"])
                        cst_base_u_all.append(result["cst_base_u"])
                        cst_base_l_all.append(result["cst_base_l"])
                        cst_le_u_all.append(result["cst_le_u"])
                        cst_le_l_all.append(result["cst_le_l"])
                        airfoil_params_all.append(result["airfoil_params"])
                    else:
                        print(f"跳过无效记录 {custom_id}（存在空字段）")
            except Exception as e:
                print(f"处理样本 {custom_id} 时出错: {e}")
                
    print(f"成功处理 {len(texts_all)} 个样本")
    
    # 使用HDF5格式保存数据
    with h5py.File(output_h5, 'w') as h5f:
        # 创建组以组织数据
        cst_group = h5f.create_group('cst')
        text_group = h5f.create_group('text')
        params_group = h5f.create_group('params')
        
        # 保存CST参数 (数值数组)
        cst_group.create_dataset('base_u', data=np.array(cst_base_u_all, dtype=np.float32))
        cst_group.create_dataset('base_l', data=np.array(cst_base_l_all, dtype=np.float32))
        cst_group.create_dataset('le_u', data=np.array(cst_le_u_all, dtype=np.float32))
        cst_group.create_dataset('le_l', data=np.array(cst_le_l_all, dtype=np.float32))
        
        # 保存文本数据 (变长字符串)
        dt = h5py.special_dtype(vlen=str)
        text_dset = text_group.create_dataset('full_text', shape=(len(texts_all),), dtype=dt)
        for i, text in enumerate(texts_all):
            text_dset[i] = text
            
        short_dset = text_group.create_dataset('short_text', shape=(len(short_descriptions_all),), dtype=dt)
        for i, desc in enumerate(short_descriptions_all):
            short_dset[i] = desc
        
        # 保存airfoil参数 (转为JSON字符串)
        params_dset = params_group.create_dataset('airfoil', shape=(len(airfoil_params_all),), dtype=dt)
        for i, params in enumerate(airfoil_params_all):
            params_dset[i] = json.dumps(params)
        
        # 记录数据集元信息
        h5f.attrs['sample_count'] = len(texts_all)
        h5f.attrs['creation_date'] = np.string_(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    print(f"数据集已保存至 {output_h5}")

def test_h5_dataset(h5_path, num_samples=5):
    """
    测试HDF5数据集完整性，随机打印几条样本进行检查
    """
    try:
        # 加载数据集
        with h5py.File(h5_path, 'r') as f:
            # 获取样本总数
            total_samples = f.attrs['sample_count']
            print(f"数据集共包含 {total_samples} 个样本")
            
            # 检查各个数组的长度是否一致
            cst_base_u = f['cst/base_u'][:]
            cst_base_l = f['cst/base_l'][:]
            cst_le_u = f['cst/le_u'][:]
            cst_le_l = f['cst/le_l'][:]
            texts = f['text/full_text'][:]
            short_texts = f['text/short_text'][:]
            params = f['params/airfoil'][:]
            
            lengths = {
                'cst_base_u': len(cst_base_u),
                'cst_base_l': len(cst_base_l),
                'cst_le_u': len(cst_le_u),
                'cst_le_l': len(cst_le_l),
                'texts': len(texts),
                'short_texts': len(short_texts),
                'params': len(params)
            }
            print(f"各字段样本数量: {lengths}")
            
            if len(set(lengths.values())) != 1:
                print("警告: 各字段样本数量不一致!")
            
            # 随机抽取样本索引
            sample_count = min(lengths.values())
            if sample_count < num_samples:
                num_samples = sample_count
                print(f"样本总数少于请求数量，将显示全部 {num_samples} 个样本")
            
            sample_indices = random.sample(range(sample_count), num_samples)
            
            # 打印随机样本
            for idx, sample_idx in enumerate(sample_indices):
                print(f"\n===== 样本 {idx+1}/{num_samples} (索引: {sample_idx}) =====")
                
                # 文本描述
                text = texts[sample_idx]
                print(f"文本描述 (前200字符): {text[:200]}..." if len(text) > 200 else text)
                
                # 短文本描述
                short_desc = short_texts[sample_idx]
                print(f"短文本描述: {short_desc}")
                
                # CST参数
                print(f"CST参数: ")
                print(f"  - 基础上表面 (10维): {cst_base_u[sample_idx]}")
                print(f"  - 基础下表面 (10维): {cst_base_l[sample_idx]}")
                print(f"  - 前缘修型上表面 (3维): {cst_le_u[sample_idx]}")
                print(f"  - 前缘修型下表面 (3维): {cst_le_l[sample_idx]}")
                
                # 翼型参数
                airfoil_params = json.loads(params[sample_idx])
                print(f"翼型参数: ")
                pprint(airfoil_params)
                
    except Exception as e:
        print(f"测试数据集时出错: {e}")

if __name__ == "__main__":
    main()
    # 替换为你的数据集路径
    h5_path = "./CLIP_dataset_cst_enhanced_PY_mix_large.h5"
    test_h5_dataset(h5_path, num_samples=3)