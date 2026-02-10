from openai import OpenAI
import os
import base64
import argparse
import pandas as pd
import json
import re
from dotenv import load_dotenv  # 导入dotenv库

# 加载环境变量
load_dotenv("/hsk/.env")

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# 初始化OpenAI客户端
client = OpenAI(
    api_key = os.getenv("AGICTO_KEY"),
    base_url = os.getenv("AGICTO_URL"),
)

prompt_1 = """ 
Use the input image and the editing prompt to create the checklist used to judge the image edited on it(mandatory).  
(Note: The editing prompt is related to unreasonable scene that against the real world)
1. VIOLATED PHYSIC LAW: Which physic law is related to the editing prompt?
2. INVOLVED OBJECTS: What are the main objects interacted with the editing prompt within the image?
3. EXPECTED PHENOMENA: What is the expected editing result of the editing prompt with respect to the involved objects(detailed and precise phenomena of the involved objects)?

Guidelines for checklist creation: 
- only target things which are visually observable in the image 
- the statements in checklist needs to be assertive statements instead of questions
- only checklist no other content
"""

def generate_checklist(model, editing_prompt, input_image):
    messages=[
    {
        "role": "user",
        "content": [
            {"type":"text","text":f"From my evaluation of editing models I have generated a image using the prompt: {editing_prompt}"},
            {"type":"text","text":"here is the input image"},
            {
                "type":"image_url",
                "image_url":{"url": f"data:image/png;base64,{input_image}"},
            },
            {"type": "text", "text": prompt_1},
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
    
    full_message = completion.choices[0].message
    answer_content = full_message.content or ""
    
    return answer_content


rubric_text = """
You are strict VLM-Judge objectively evaluating a generated edited image from a checklist, editing prompt and the input image. 
The checklist is provided as a reference :
1. VIOLATED PHYSIC LAW
2. INVOLVED OBJECTS
3. EXPECTED PHENOMENA

Score each rubric from 1–4 (1=absent/contradictory, 2=weak/partly wrong, 3=mostly correct, 4=clearly correct): 
a) prompt consistency — follows instructions: correct setup and correct experiment execution. 
b) expected phenomenon — expected physical/chemical outcome is present according to the EXPECTED PHENOMENA(check the image in detail).
c) involved objects coherence — whether the involved objects are the same object between the input image and the edited image beside some changes according to the EXPECTED PHENOMENA. 
d) background immutability — background objects remain intact/unchanged unless changes are explicitly expected,which should be continious with the involved objects. 

Each rating must be supported with clear justification, drawing on specific edited image area and, when provided, the corresponding checklist items.
"""
output_format = """
Return JSON with fields: 
{ ”scores”: { ”prompt_consistency”:1-4, ”expected_phenomenon”:1-4, ”involved_objects_coherence”:1-4, ”background_immutability”:1-4}, 
 ”explanations”: {”summary”: string, ”issues”: [{"issue_name": string,"score_explanation":string}...]} ## issue_name's value should one of: prompt consistency, expected phenomenon, involved objects coherence and background immutability
 }
"""

# 
def vlm_judge(editing_prompt, edited_image, input_image, check_list, model_name):
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here is the editing prompt :{editing_prompt}"},
                {"type": "text", "text": f"here is the input image:"},
                {
                    "type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{input_image}"},
                },
                {"type": "text", "text": f"here is the edited image:"},
                {
                    "type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{edited_image}"},
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
)

    full_message = completion.choices[0].message
    score_content = full_message.content or ""
    
    return score_content


def parse_score_content(score_content):
    """
    解析评分内容，提取四个评分和解释
    """
    try:
        # 尝试直接解析JSON
        score_data = json.loads(score_content)
        scores = score_data.get("scores", {})
        explanations = score_data.get("explanations", {})
        
        # 提取四个评分 - 使用正确的键名
        prompt_consistency = scores.get("prompt_consistency")
        expected_phenomenon = scores.get("expected_phenomenon")  # 修正：使用正确的键名
        involved_objects_coherence = scores.get("involved_objects_coherence")
        background_immutability = scores.get("background_immutability")
        
        # 提取解释
        summary = explanations.get("summary", "")
        issues = explanations.get("issues", [])
        
        return {
            "prompt_consistency": prompt_consistency,
            "expected_phenomenon": expected_phenomenon,
            "involved_objects_coherence": involved_objects_coherence,
            "background_immutability": background_immutability,
            "summary": summary,
            "issues": issues
        }
    except json.JSONDecodeError:
        # 如果不是有效的JSON，尝试使用正则表达式解析
        parsed_data = {}

        # 提取评分 - 查找各个评分的值
        pc_match = re.search(r'"prompt consistency"\s*:\s*(\d)', score_content)
        ep_match = re.search(r'"expected phenomenon"\s*:\s*(\d)', score_content)
        ioc_match = re.search(r'"involved objects coherence"\s*:\s*(\d)', score_content)
        bi_match = re.search(r'"background immutability"\s*:\s*(\d)', score_content)

        if pc_match:
            parsed_data["prompt_consistency"] = int(pc_match.group(1))
        else:
            parsed_data["prompt_consistency"] = None

        if ep_match:
            parsed_data["expected_phenomenon"] = int(ep_match.group(1))
        else:
            parsed_data["expected_phenomenon"] = None

        if ioc_match:
            parsed_data["involved_objects_coherence"] = int(ioc_match.group(1))
        else:
            parsed_data["involved_objects_coherence"] = None

        if bi_match:
            parsed_data["background_immutability"] = int(bi_match.group(1))
        else:
            parsed_data["background_immutability"] = None

        # 提取解释部分 - 这可能需要根据实际返回格式调整
        summary_match = re.search(r'"summary":\s*"([^"]*)"', score_content)
        if summary_match:
            parsed_data["summary"] = summary_match.group(1)
        else:
            parsed_data["summary"] = ""

        # 尝试找到issues部分
        issues_matches = re.findall(r'\{\s*"issue_name":\s*"([^"]+)",\s*"score_explanation":\s*"([^"]*)"', score_content)
        parsed_data["issues"] = []
        for issue_match in issues_matches:
            parsed_data["issues"].append({
                "issue_name": issue_match[0],
                "score_explanation": issue_match[1]
            })

        return parsed_data


def load_existing_data(file_path):
    """从JSONL文件加载已有数据"""
    data_dict = {}
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 确保行不为空
                    try:
                        data = json.loads(line)
                        data_id = data.get('data_id')
                        if data_id is not None:
                            data_dict[data_id] = data
                    except json.JSONDecodeError as e:
                        print(f"警告：无法解析JSON行: {line}, 错误: {e}")
    return data_dict


def append_to_jsonl(data, file_path):
    """向JSONL文件追加一行数据"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')


def process_dataset(gemini_gen_path, output_base_path, model_name, target_model_folder):
    # 创建输出目录（用于存放模型输出图像）
    output_model_path = os.path.join(output_base_path, target_model_folder)
    os.makedirs(output_model_path, exist_ok=True)
    
    # 加载meta.jsonl文件
    meta_file_path = os.path.join(gemini_gen_path, "meta.jsonl")
    
    # 输出文件路径：所有模型共享一个checklist，但每个模型有自己的score文件
    checklist_output_path = os.path.join(output_base_path, "checklists.jsonl")
    score_output_path = os.path.join(output_model_path, "scores.jsonl")  # score保存在各自模型目录中
    
    # 加载已经处理过的checklist和score
    processed_checklists = load_existing_data(checklist_output_path)
    processed_scores = load_existing_data(score_output_path)

    # 读取meta.jsonl
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            data_id = data['data_id']
            edit_prompt = data['edit_prompt']
            
            print(f"Processing data_id: {data_id}")
            
            # 检查input_image是否存在
            input_image_path = os.path.join(gemini_gen_path, f"input_data/data_{data_id}.png")
            
            if not os.path.exists(input_image_path):
                print(f"Input image does not exist for data_id {data_id}: {input_image_path}")
                continue
            
            # 生成checklist（如果没有的话）
            if data_id not in processed_checklists:
                print(f"Generating checklist for data_id {data_id}")
                
                input_image_b64 = encode_image(input_image_path)
                checklist = generate_checklist(model_name, edit_prompt, input_image_b64)
                
                # 保存checklist
                checklist_data = {
                    "data_id": data_id,
                    "data_type": data['data_type'],
                    "sub_id": data['sub_id'],
                    "checklist": checklist
                }
                
                append_to_jsonl(checklist_data, checklist_output_path)
                
                processed_checklists[data_id] = checklist_data
                print(f"Saved checklist for data_id {data_id}")
            else:
                print(f"Checklist already exists for data_id {data_id}")
            
            # 检查output_image是否存在
            output_image_path = os.path.join(output_model_path, f"output_{data_id}.png")
            
            if not os.path.exists(output_image_path):
                print(f"Output image does not exist for data_id {data_id}: {output_image_path}, skipping score generation")
                continue
            
            # 生成score（如果output图像存在且还没有score的话）
            if data_id not in processed_scores:
                print(f"Generating score for data_id {data_id}")
                
                # 获取checklist（可能刚生成或之前已存在）
                checklist = processed_checklists[data_id]['checklist']
                
                input_image_b64 = encode_image(input_image_path)
                output_image_b64 = encode_image(output_image_path)
                
                score_content = vlm_judge(edit_prompt, output_image_b64, input_image_b64, checklist, model_name)
                
                # 解析评分内容
                parsed_scores = parse_score_content(score_content)
                
                # 保存解析后的评分
                score_data = {
                    "data_id": data_id,
                    "data_type": data['data_type'], 
                    "sub_id": data['sub_id'],
                    "prompt_consistency": parsed_scores["prompt_consistency"],
                    "expected_phenomenon": parsed_scores["expected_phenomenon"],  # 修正字段名拼写
                    "involved_objects_coherence": parsed_scores["involved_objects_coherence"],
                    "background_immutability": parsed_scores["background_immutability"],
                    "summary": parsed_scores["summary"],
                    "issues": parsed_scores["issues"]
                }
                
                append_to_jsonl(score_data, score_output_path)
                
                processed_scores[data_id] = score_data
                print(f"Saved parsed scores for data_id {data_id}")
            else:
                print(f"Score already exists for data_id {data_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate checklist and scores for image editing evaluation.')
    parser.add_argument('--gemini-gen-path', type=str, default='/hsk/dataset_test/anti-physic/gemini_gen_antiphysic', 
                        help='Path to the gemini_gen_antiphysic folder')
    parser.add_argument('--output-base-path', type=str, default='/hsk/dataset_test/anti-physic/output_file', 
                        help='Base path to the output files folder')
    parser.add_argument('--model-name', type=str, default="gemini-3-pro-preview-thinking", 
                        help='Model used for generating checklists and scores')
    parser.add_argument('--target-model-folder', type=str, required=True, 
                        help='Target model folder name in output_base_path')
    
    args = parser.parse_args()
    
    process_dataset(args.gemini_gen_path, args.output_base_path, args.model_name, args.target_model_folder)