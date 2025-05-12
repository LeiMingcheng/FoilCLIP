import json
import re

# 输入和输出文件路径
input_file = "./json_data/airfoils_requests_generated.jsonl"
output_file = "./json_data/airfoils_requests_generated_en.jsonl"

# 中文到英文的映射
zh_to_en_mapping = {
    # 几何描述字符串模式替换
    r"最大厚度 (.*?) 在 (.*?)，最大弯度 (.*?) 在 (.*?)": 
        r"Maximum thickness \1 at \2, maximum camber \3 at \4",
    
    # 系统提示替换
    "你是一个翼型专家，正在分析一个翼型，请你以仿佛*看得见*的视角，简要描述这个翼型，重点强调几何与气动特性（例如低弯度、高升力等），最终描述不超过100字。": 
        "You are an airfoil expert analyzing an airfoil. As if you can *see* it, briefly describe this airfoil, emphasizing geometric and aerodynamic characteristics (e.g., low camber, high lift, etc.). Keep the description under 100 words."
}

# 处理函数
def process_jsonl():
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        line_count = 0
        modified_count = 0
        
        for line in f_in:
            line_count += 1
            data = json.loads(line.strip())
            
            # 检查并替换系统消息内容
            if "body" in data and "messages" in data["body"]:
                for message in data["body"]["messages"]:
                    if message["role"] == "system" and message["content"] in zh_to_en_mapping:
                        message["content"] = zh_to_en_mapping[message["content"]]
                        modified_count += 1
            
            # 检查并替换用户消息中的几何描述字符串
            if "body" in data and "messages" in data["body"]:
                for message in data["body"]["messages"]:
                    if message["role"] == "user":
                        for pattern, replacement in zh_to_en_mapping.items():
                            # 只针对正则表达式模式进行替换
                            if pattern.startswith(r"最大厚度"):
                                message["content"] = re.sub(pattern, replacement, message["content"])
                                if re.search(pattern, message["content"]):
                                    modified_count += 1
            
            # 写入修改后的行
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"处理完成：共处理 {line_count} 行，修改 {modified_count} 处内容")

# 新函数：在jsonl文件的user消息结尾添加指定文本
def add_sentence_order_note():
    jsonl_file = "./json_data/airfoils_requests_generated_en.jsonl"  # 输入文件
    output_file = "./json_data/airfoils_requests_generated_en_modified.jsonl"  # 创建新文件而不是覆盖
    
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        line_count = 0
        modified_count = 0
        
        for line in f_in:
            line_count += 1
            data = json.loads(line.strip())
            
            # 在用户消息结尾添加文本
            if "body" in data and "messages" in data["body"]:
                for message in data["body"]["messages"]:
                    if message["role"] == "user" and not message["content"].endswith("(The sentence order can be modified freely)"):
                        message["content"] += " (The sentence order can be modified freely)"
                        modified_count += 1
            
            # 写入修改后的行
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"处理完成：共处理 {line_count} 行，修改 {modified_count} 处用户消息")
    
    print(f"已创建新文件: {output_file}")

if __name__ == "__main__":
    # process_jsonl()  # 注释掉原函数调用
    add_sentence_order_note()  # 调用新函数