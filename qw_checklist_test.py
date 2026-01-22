
from openai import OpenAI
import os
import base64
import argparse
import pandas as pd
import json


#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# 初始化OpenAI客户端
client = OpenAI(
    api_key = "sk-b09c374d8bd2478fa94697ae79dad1bd",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


prompt = """ 
Use reference source to create the checklist used to judge the image edited on it(mandatory).  
Create a comprehensive checklist targeting the following categories (note: not all categories are required for a prompt). 
1. PHENOMENON CONGRUENCY: Does the image show the correct expected phenomenon?  
2. CORRECT DYNAMISM Are the physics dynamics and motion behaviors accurate?  
3. SPATIO-TEMPORAL CONTINUITY Are spatial relationships and temporal sequences physically consistent?  
4. IMMUTABILITY Do object properties remain physically consistent?  
5. INTERACTION REALISM Do object interactions follow physical laws?  
Guidelines for checklist creation: 
- only target things which are visually observable in the image 
- the statements in checklist needs to be assertive statements instead of questions
"""

def generate_checklist(prompt, model,editing_prompt, reference_phenomenon,input_image, gt_image):
    messages=[
    {
        "role": "user",
        "content": [
            {"type":"text","text":f"From my evaluation of editing models I have generated a image using the prompt Prompt: {editing_prompt}"},
            {"type":"text","text":"here is the input image"},
            {
                "type":"image_url",
                "image_url":{"url": f"data:image/png;base64,{input_image}"},
            },
            {"type":"text","text":"here is the ground-true image"},
            {
                "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gt_image}"},
            },
            {"type":"text","text":f"reference phenomenon:{reference_phenomenon}"},
            {"type": "text", "text": prompt},
        ],
    },
]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},
    )
    if hasattr(completion.choices[0], 'message'):
    # 获取完整消息内容
        full_message = completion.choices[0].message
    
    # 检查是否存在思考内容
    # if hasattr(full_message, 'reasoning_content'):
    #     reasoning_content = full_message.reasoning_content or ""
        
    #     if enable_thinking:
    #         print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    #         print(reasoning_content)
    
    # 获取实际的答案内容
    answer_content = full_message.content or ""
    
    
    return answer_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate checklist for image editing evaluation.')
    parser.add_argument('--dataset-dir-path', type=str, required=True, help='Path to the input and ground truth image')
    parser.add_argument('--output-file', type=str, default="checklist.json", help='Output file to save the answer content')
    parser.add_argument('--model-name', type=str, default="qwen3-vl-235b-a22b-thinking", help='model used to generate the checklist')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir_path
    prompt_data_path = os.path.join(dataset_dir, "prompt.csv")
    prompt_data = pd.read_csv(prompt_data_path)
    check_lists = {}
    
    for i,item in enumerate(os.listdir(dataset_dir)):
        if os.path.isdir(os.path.join(dataset_dir,item)):
            print(f"processing {item}")
            index = int(item[-1])
            test_dir = os.path.join(dataset_dir, f"{item}")
            input_image = encode_image(os.path.join(test_dir, "input.png"))
            gt_image = encode_image(os.path.join(test_dir, "gt.png"))
            editing_prompt = prompt_data.iloc[index-1]["editing_prompt"]
            print(f"editing prompt: {editing_prompt[:15]}")
            reference_phenomenon = prompt_data.iloc[index-1]["reference_phenomenon"]
            print(f"reference phenomenon: {reference_phenomenon[:15]}")
            check_list = generate_checklist(prompt, args.model_name,editing_prompt, reference_phenomenon,input_image, gt_image)
            check_lists[f"{item}"] = check_list
    json.dump(check_lists, open(os.path.join(dataset_dir,args.output_file), "w"), indent=4)
        

    
    


    
   