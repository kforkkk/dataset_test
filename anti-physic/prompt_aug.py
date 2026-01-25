import os
import json
import re
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key = "sk-b09c374d8bd2478fa94697ae79dad1bd",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def augument_prompt(prompt):
    sys_aug = """
You are good at transfering the origin scene prompts to a editing prompt by adding some static object and change the syntactical structure. 
There are several requirements:
    1. Remember only at most 3 background-content-related objects is allowed to be added to the prompt.
    2. you should pay attention that the origin prompt has unreasonable scene. 
Follow these steps:
First, reason about the formal reasonable scene temporally before the unreasonable scene in the prompt. 
Then, transfer the oringin prompt to a editing prompt,conditioned on the formal scene, which depict the process from the formal scene to the unreasonable scene .

Output format:

formal_reasonable_scene:
...
editing_prompt:
...
"""
    messages = [
        {"role": "system", "content": sys_aug},
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model="qwen-plus",
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
    print("answer_content:",answer_content)
    # 使用更精确的正则表达式分别提取formal reasonale scene和editing prompt
    # 匹配"formal"相关的字段，并确保不会匹配到editing相关内容
    formal_pattern = r'(?:formal[_\s-]?reasonable?[_\s-]?scene?|formal[_\s-]?scene?|reasonable?[_\s-]?scene?):\s*(.*?)(?=\n\s*(?:editing[_\s-]?prompt|editing|modify|revise):\s*|$)'
    # 匹配"editing prompt"相关的字段
    editing_pattern = r'(?:editing[_\s-]?prompt|editing|modify|revise):\s*(.*?)(?:\n\s*(?:formal|output|result|answer)[_\s-]*:|$)'
    
    formal_match = re.search(formal_pattern, answer_content, re.IGNORECASE | re.DOTALL)
    editing_match = re.search(editing_pattern, answer_content, re.IGNORECASE | re.DOTALL)
    
    formal_scene = formal_match.group(1).strip() if formal_match else ""
    editing_pr = editing_match.group(1).strip() if editing_match else ""
    
    return formal_scene, editing_pr

root_dir = "/hsk/dataset_test/anti-physic"  # 设置正确的根目录路径
for item in tqdm(os.listdir(root_dir)):
    if os.path.isdir(os.path.join(root_dir, item)):
        data_path = os.path.join(root_dir, item, f"{item}.jsonl")
        
        # 读取现有的jsonl文件
        updated_lines = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data = json.loads(line)
                    origin_prompt = data['prompt']
                    print("origin_prompt:",origin_prompt)
                    
                    # 删除旧的aug_prompt字段（如果存在）
                    if 'aug_prompt' in data:
                        del data['aug_prompt']
                    
                    formal_scene, editing_pr = augument_prompt(origin_prompt)
                    data['formal_reasonable_scene'] = formal_scene
                    data['editing_prompt'] = editing_pr
                    # print("formal_reasonable_scene:",formal_scene)
                    # print("editing_prompt:",editing_pr)
                    
                    # 将更新后的数据转换为json字符串并添加到列表
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
        
        # 将更新后的数据写回到文件
        with open(data_path, 'w', encoding='utf-8') as f:
            for line in updated_lines:
                f.write(line + '\n')