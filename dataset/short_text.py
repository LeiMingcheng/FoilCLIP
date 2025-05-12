import json
import re

def extract_airfoil_descriptions(jsonl_file_path):
    """从JSONL文件中提取翼型描述内容"""
    descriptions = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                # 从JSON中提取content内容
                content = record["response"]["body"]["choices"][0]["message"]["content"]
                #content = record["response"]["body"]["content"]
                descriptions.append(content)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"解析记录时出错: {e}")
                continue
    
    return descriptions



def generate_short_description_prompt(original_record):
    """构建符合原结构的请求记录"""
    #cst_params = original_record["response"]["body"]["cst_parameters"]
    #airfoil_params = original_record["response"]["body"]["airfoil_params"]
    
    prompt_text = f"""As an airfoil expert, please generate a brief one-sentence summary (no more than 20 words) of the following airfoil description, highlighting its most critical features.

Detailed description:
{original_record["response"]["body"]["choices"][0]["message"]["content"]}

Please output a short sentence without any explanation.
Output examples:
-Supercritical airfoil.
-Natural laminar flow airfoil.
"""
    
    return {
        "custom_id": original_record["custom_id"],
        #"cst_parameters": cst_params,
        #"airfoil_parameters": airfoil_params,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-v3",
            "temperature": 0.5,  # 降低随机性保证稳定性
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an aviation airfoil expert, skilled at summarizing airfoil core features in one sentence"
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        }
    }

import json



def process_jsonl_file(input_file, output_file):
    """生成符合原结构的请求文件"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                original_record = json.loads(line)
                request_record = generate_short_description_prompt(original_record)
                outfile.write(json.dumps(request_record, ensure_ascii=False) + '\n')
            
            except (KeyError, json.JSONDecodeError) as e:
                print(f"处理记录时出错: {e}")
                continue

# 示例使用
i = 4
process_jsonl_file(f"json_data/descriptions/split/foil_description_{i}.jsonl", f"json_data/requests/short_requests/short_request_{i}.jsonl")
