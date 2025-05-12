import json
import os
import argparse
from pathlib import Path

def split_jsonl_file(input_file, output_dir, max_lines=40000):
    """
    将 JSONL 文件拆分为多个小文件，每个文件最多包含指定行数的 JSON 记录
    
    参数:
        input_file: 输入的 JSONL 文件路径
        output_dir: 输出目录
        max_lines: 每个输出文件的最大行数
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入文件名（不带扩展名）
    base_name = Path(input_file).stem
    
    # 初始化计数器
    file_index = 0
    line_count = 0
    
    # 打开当前输出文件
    output_file = open(os.path.join(output_dir, f"{base_name}_{file_index}.jsonl"), 'w', encoding='utf-8')
    
    # 逐行读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 检查是否需要创建新文件
            if line_count >= max_lines:
                output_file.close()
                file_index += 1
                line_count = 0
                output_file = open(os.path.join(output_dir, f"{base_name}_{file_index}.jsonl"), 'w', encoding='utf-8')
            
            # 写入当前行
            output_file.write(line)
            line_count += 1
    
    # 关闭最后一个输出文件
    output_file.close()
    
    print(f"拆分完成！共生成 {file_index + 1} 个文件，保存在 {output_dir} 目录中")

def merge_jsonl_files(input_dir, output_file):
    """
    将指定目录中的所有 JSONL 文件合并为一个大文件
    
    参数:
        input_dir: 包含要合并的 JSONL 文件的目录
        output_file: 合并后的输出文件路径
    """
    # 创建输出文件所在的目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 获取目录中的所有 JSONL 文件并按名称排序
    jsonl_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jsonl')])
    
    # 记录处理的文件数和记录数
    total_files = len(jsonl_files)
    total_records = 0
    
    # 如果没有找到 JSONL 文件
    if total_files == 0:
        print(f"在 {input_dir} 中没有找到 JSONL 文件")
        return
    
    # 创建输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 逐个处理每个输入文件
        for i, filename in enumerate(jsonl_files):
            file_path = os.path.join(input_dir, filename)
            file_records = 0
            
            print(f"正在处理 [{i+1}/{total_files}]: {filename}")
            
            # 读取当前文件并写入输出文件
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    file_records += 1
            
            total_records += file_records
            print(f"  - 已合并 {file_records} 条记录")
    
    print(f"合并完成！总共合并了 {total_files} 个文件，共 {total_records} 条记录。")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 定义拆分功能的参数
    #input_file = 'json_data/requests/airfoils_requests_generated_en.jsonl'
    #output_dir = 'json_data/requests/airfoils_requests_split'
    max_lines = 45000
    
    # 定义合并功能的参数
    merge_input_dir = 'json_data/descriptions/short_text'
    merge_output_file = 'json_data/descriptions/short_text.jsonl'
    
    # 取消注释下面的行来执行拆分操作
    #split_jsonl_file(input_file, output_dir, max_lines)
    
    # 取消注释下面的行来执行合并操作
    merge_jsonl_files(merge_input_dir, merge_output_file)