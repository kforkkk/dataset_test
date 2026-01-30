import os
import json
import shutil
from tqdm import tqdm

root = "/hsk/dataset_test/anti-physic"
output_dir = "/hsk/dataset_test/anti-physic/input_data"
os.makedirs(output_dir, exist_ok=True)

num = 1
meta_entries = []

# 读取每个类别的jsonl文件，获取input_prompt
jsonl_data = {}
for item in os.listdir(root):
    jsonl_path = os.path.join(root, item, f"{item}.jsonl")
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            jsonl_data[item] = [json.loads(line) for line in lines]

for item in tqdm(os.listdir(root), desc="processing"):
    item_path = os.path.join(root, item)
    if os.path.isdir(item_path) and item!="input_data":
        png_count = 0  # 记录当前类别中的图片序号
        
        for stuff in os.listdir(item_path):
            if stuff.endswith("gemini.png"):
                old_path = os.path.join(item_path, stuff)
                # 创建新的文件名，保持.png扩展名
                new_filename = f"data_{num}.png"
                new_path = os.path.join(output_dir, new_filename)
                
                # 复制文件而不是重命名
                shutil.copy2(old_path, new_path)
                print(f"已将 {old_path} 复制到 {new_path}")
                
                # 获取对应的input_prompt
                # 根据文件名提取索引 (例如: category_origin_2-gemini.png -> 索引为2)
                import re
                match = re.search(r'_origin_(\d+)-gemini\.png$', stuff)
                input_prompt = ""
                if match and item in jsonl_data:
                    idx = int(match.group(1))
                    if idx < len(jsonl_data[item]):
                        input_prompt = jsonl_data[item][idx]["input_prompt"]
                
                # 记录元数据：总体顺序、所属类别、类别中的顺序、输入提示和编辑提示
                meta_entry = {
                    "data_id": num,           # 图片总体顺序
                    "data_type": item,        # 属于哪个类别
                    "sub_id": png_count + 1,  # 类别中的顺序
                    "input_prompt": input_prompt,  # 输入提示
                    "edit_prompt": ""         # 编辑提示（空）
                }
                meta_entries.append(meta_entry)
                
                num += 1
                png_count += 1

# 将元数据写入meta.jsonl文件
meta_output_path = os.path.join(root, "meta.jsonl")
with open(meta_output_path, "w", encoding="utf-8") as f:
    for meta_entry in meta_entries:
        f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

print(f"处理完成！共处理 {len(meta_entries)} 张图片。")
print(f"所有图片已保存到 {output_dir} 目录下")
print(f"所有元数据已保存到 {meta_output_path}")