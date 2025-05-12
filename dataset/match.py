import json
import os
FOIL1 = "./json_data/requests/airfoils_requests_generated_en.jsonl"
FOIL2 = "./json_data/descriptions/foil_description_generated.jsonl"
FOIL2_WITH_CST = "./json_data/descriptions/foil_description_generated_with_cst.jsonl"

# 检查文件是否存在
def match_cst():
    if not os.path.exists(FOIL1):
        print(f"错误：找不到文件 '{FOIL1}'")
        exit(1)
    if not os.path.exists(FOIL2):
        print(f"错误：找不到文件 '{FOIL2}'")
        exit(1)

    # 读取第一个文件中的CST参数和翼型参数
    cst_params = {}
    airfoil_params = {}
    
    try:
        with open(FOIL1, 'r', encoding='utf-8') as f1:
            for line in f1:
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    
                    # 检查CST参数
                    if custom_id and 'body' in data and 'cst_parameters' in data['body']:
                        cst_params[custom_id] = data['body']['cst_parameters']
                    
                    # 检查翼型参数
                    if custom_id and 'body' in data and 'airfoil_parameters' in data['body']:
                        airfoil_params[custom_id] = data['body']['airfoil_parameters']
                        
                except json.JSONDecodeError as e:
                    print(f"解析第一个文件中的JSON时出错: {e}")
                    continue
        
        print(f"从第一个文件中读取了 {len(cst_params)} 个CST参数和 {len(airfoil_params)} 个翼型参数")
    except Exception as e:
        print(f"读取第一个文件时出错: {e}")
        exit(1)

    # 读取第二个文件并添加CST参数和翼型参数
    updated_count = 0
    try:
        with open(FOIL2, 'r', encoding='utf-8') as f2:
            updated_lines = []
            for line_num, line in enumerate(f2, 1):
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    
                    # 初始化更新标志
                    updated = False
                    
                    # 确保response和body结构存在
                    if 'response' not in data:
                        data['response'] = {}
                    if 'body' not in data['response']:
                        data['response']['body'] = {}
                    
                    # 如果在第一个文件中找到匹配的custom_id，则添加CST参数
                    if custom_id in cst_params:
                        data['response']['body']['cst_parameters'] = cst_params[custom_id]
                        updated = True
                    
                    # 添加翼型参数
                    if custom_id in airfoil_params:
                        data['response']['body']['airfoil_parameters'] = airfoil_params[custom_id]
                        updated = True
                    
                    if updated:
                        updated_count += 1
                    
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError as e:
                    print(f"解析第二个文件第{line_num}行时出错: {e}")
                    updated_lines.append(line.strip())  # 保留原始行
                    continue
        
        print(f"更新了 {updated_count} 条记录")
    except Exception as e:
        print(f"读取第二个文件时出错: {e}")
        exit(1)

    # 写入更新后的文件
    try:
        with open(FOIL2_WITH_CST, 'w', encoding='utf-8') as f_out:
            for line in updated_lines:
                f_out.write(line + '\n')
        
        print(f"处理完成！成功写入 '{FOIL2_WITH_CST}'")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")

# 读取第一个文件中的CST参数和翼型参数
def match_short_text():
    short_text_file = 'json_data/descriptions/short_text_0.jsonl'
    foil_description_file = 'json_data/descriptions/foil_description_base_with_cst.jsonl'
    cst_params = {}
    airfoil_params = {}
    short_text = {}
    try:
        with open(short_text_file, 'r', encoding='utf-8') as f1:
            for line in f1:
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    if custom_id:
                        short_text[custom_id] = data['response']['body']['choices'][0]['message']['content']
                except json.JSONDecodeError as e:
                    print(f"解析第一个文件中的JSON时出错: {e}")
                    continue
        print(f"从第一个文件中读取了 {len(short_text)} 个简短描述")
    except Exception as e:
        print(f"读取第一个文件时出错: {e}")
        exit(1)

    # 读取第二个文件并添加CST参数和翼型参数
    updated_count = 0
    try:
        with open(foil_description_file, 'r', encoding='utf-8') as f2:
            updated_lines = []
            for line_num, line in enumerate(f2, 1):
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    
                    if custom_id in short_text:
                        data['response']['body']['short_text'] = short_text[custom_id]
                    updated_count += 1
                    
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError as e:
                    print(f"解析第二个文件第{line_num}行时出错: {e}")
                    updated_lines.append(line.strip())  # 保留原始行
                    continue
        
        print(f"更新了 {updated_count} 条记录")
    except Exception as e:
        print(f"读取第二个文件时出错: {e}")
        exit(1)

    # 写入更新后的文件
    try:
        with open(foil_description_file, 'w', encoding='utf-8') as f_out:
            for line in updated_lines:
                f_out.write(line + '\n')
        
        print(f"处理完成！成功写入 '{foil_description_file}'")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")

if __name__ == "__main__":
    #match_cst()
    match_short_text()