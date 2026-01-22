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

rubric_text = """
You are VLM-Judge evaluating a generated edited image from a ground true image and its reference phenomenon. 
Score each rubric from 1–4 (1=absent/contradictory, 2=weak/partly wrong, 3=mostly correct, 4=clearly correct): 
a) prompt consistency — follows instructions: correct setup and correct experiment execution. 
b) expected phenomenon — expected physical/chemical outcome is present and correct. 
c) immutability — objects remain intact/unchanged unless changes are explicitly expected. 
d) dynamism — other physical laws are obeyed. 
e) coherence — natural transitions across images; no flicker/teleport/identity swap.  
Each rating must be supported with clear justification, drawing on specific edited image area and, when provided, the corresponding checklist items.
"""
output_format = """
Return JSON with fields: 
{ ”scores”: { ”prompt consistency”:1-4, ”expected phenomenon”:1-4, ”immutability”:1-4, ”dynamism”:1-4, ”coherence”:1-4 }, 
 ”explanations”: {”summary”: string, ”issues”: [string]}, 
 ”evidence”: {”candidate”: [{"area":"","observation":""},..], 
 ”reference”: [{"area":"","observation":""}]} }
"""

# 
def vlm_judge(question_description, gt_phenomenon, edited_image, gt_image, check_list,model_name):
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question description:{question_description}"},
                {"type": "text", "text": f"ground true phenomenon:{gt_phenomenon}"},
                {"type": "text", "text": f"here is the edited image:"},
                {
                    "type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{edited_image}"},
                },
                {"type": "text", "text": f"here is the ground true image:"},
                {
                    "type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{gt_image}"},
                },
                {"type": "text", "text": f"here is the checklist:{check_list}"},
                {"type": "text", "text": rubric_text},
                {"type": "text", "text": f"here is the output format:{output_format}"},
            ],
        },
    ],
    stream=False,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},

    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
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
    score_content = full_message.content or ""
    
    
    return score_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate checklist for image editing evaluation.')
    parser.add_argument('--dataset-dir-path', type=str, required=True, help='Path to the input and ground truth image')
    parser.add_argument('--output-file-name', type=str, default="score", help='Output file to save the answer content')
    parser.add_argument('--model-name', type=str, default="qwen3-vl-235b-a22b-thinking", help='model used to generate the checklist')
    parser.add_argument('--tested-model', type=str, required=True, help='model to be tested')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir_path
    prompt_data_path = os.path.join(dataset_dir, "prompt.csv")
    prompt_data = pd.read_csv(prompt_data_path)
    scores = {}
    check_lists = json.load(open(os.path.join(dataset_dir, "checklist.json"), "r"))
    for i,item in enumerate(os.listdir(dataset_dir)):
        if os.path.isdir(os.path.join(dataset_dir,item)):
            print(f"processing {item}")
            #获取文本信息
            index = int(item[-1])
            question_description = prompt_data.iloc[index-1]['editing_prompt']
            gt_phenomenon = prompt_data.iloc[index-1]['reference_phenomenon']
            test_dir = os.path.join(dataset_dir, f"{item}")
            #获取图片信息
            gt_image = encode_image(os.path.join(test_dir, "gt.png"))
            edited_image = encode_image(os.path.join(test_dir, f"{args.tested_model}-output.png"))
            check_list = check_lists[f"{item}"]
            #生成评分
            score_content = vlm_judge(question_description, gt_phenomenon, edited_image, gt_image, check_list,args.model_name)
            scores[f"{item}"] = score_content
    json.dump(scores, open(os.path.join(dataset_dir, args.output_file_name +f"-{args.tested_model}"+ ".json"), "w"),indent=4)

