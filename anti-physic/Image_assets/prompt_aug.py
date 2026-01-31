import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv("/hsk/.env")

client = OpenAI(
    api_key = os.getenv("QWEN_URL"),
    base_url=os.getenv("QWEN_KEY")
)

def enhance_formal_scene(formal_scene):
    sys_prompt = """
You are skilled at transforming plausible, realistic scenes into effective text-to-image prompts.  
Please follow these guidelines:  
1. Use a medium long shot camera view.  
2. Employ deep depth of field.  
3. Keep everything in sharp focus, with the background rendered in subtle, realistic detail.  
4. The prompt must be concise, precise, and specific, use less than 80 words.  
5. The scene must appear realistic and capture a moment just before a specific action occurs—tense, poised, but not yet executed.

Output format:

enhanced_input_prompt:
...
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"formal_resonable_scene"+formal_scene},
    ]
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        stream=False,
        extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},
    )
    if hasattr(completion.choices[0], 'message'):
    # 获取完整消息内容
        full_message = completion.choices[0].message
    
    answer_content = full_message.content or ""
    print("enhanced_content:", answer_content)
    
    # 提取增强后的输入提示
    enhanced_pattern = r'(?:enhanced[_\s-]?input[_\s-]?prompt|input[_\s-]?prompt|enhanced[_\s-]?prompt|new[_\s-]?prompt):\s*(.*?)(?:\n\s*(?:formal|output|result|answer|editing)[_\s-]*:|$)'
    
    enhanced_match = re.search(enhanced_pattern, answer_content, re.IGNORECASE | re.DOTALL)
    
    enhanced_input_prompt = enhanced_match.group(1).strip() if enhanced_match else formal_scene
    
    return enhanced_input_prompt


def process_jsonl_files(root_dir="/hsk/dataset_test/anti-physic"):
    """
    处理anti-physic中除了hunyuan这个文件夹的其他所有文件夹下的所有jsonl文件，
    对每个jsonl文件中的每一行json进行遍历，把formal_reasonable_scene这个键的值
    提取放给VLM进行增强，形成新的input_prompt这个键
    """
    for item in tqdm(os.listdir(root_dir), desc="Processing folders"):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item != "hunyuan":  # 排除hunyuan文件夹
            data_path = os.path.join(item_path, f"{item}.jsonl")
            
            # 检查jsonl文件是否存在
            if not os.path.exists(data_path):
                print(f"文件不存在: {data_path}")
                continue
                
            # 读取现有的jsonl文件
            updated_lines = []
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in tqdm(lines, desc=f"Processing {item}", leave=False):
                line = line.strip()
                if line:  # 跳过空行
                    data = json.loads(line)
                    
                    # 检查是否包含formal_reasonable_scene字段
                    if 'formal_reasonable_scene' not in data:
                        print(f"跳过没有formal_reasonable_scene字段的数据: {data.get('id', 'unknown')}")
                        updated_lines.append(json.dumps(data, ensure_ascii=False))
                        continue
                    
                    formal_scene = data['formal_reasonable_scene']
                    print("original_formal_scene:", formal_scene)
                    
                    # 删除旧的input_prompt字段（如果存在）
                    if 'input_prompt' in data:
                        del data['input_prompt']
                    
                    # 使用VLM增强formal_reasonable_scene
                    enhanced_input_prompt = enhance_formal_scene(formal_scene)
                    
                    # 添加新的input_prompt字段
                    data['input_prompt'] = enhanced_input_prompt
                    print("enhanced_input_prompt:", enhanced_input_prompt)
                    
                    # 将更新后的数据转换为json字符串并添加到列表
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
            
            # 将更新后的数据写回到文件
            with open(data_path, 'w', encoding='utf-8') as f:
                for line in updated_lines:
                    f.write(line + '\n')


# 如果直接运行此脚本，则执行处理
if __name__ == "__main__":
    process_jsonl_files()