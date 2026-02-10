import re
import argparse
import shutil
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import json
from tqdm import tqdm

load_dotenv("/hsk/.env")

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("CHATANYWHERE_KEY"),
    base_url=os.getenv("CHATANYWHERE_URL")
    
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def frame_selection(caption, frame_dir):
    sys = f"""

The image displays the source photo at the top, with a collage of 12 edited versions beneath it. 
The target edit image caption was: {caption}. Your task is to choose the image from 1 to 12 that best follows this edit fully and naturally. 
If none of the images follows the edit, select image 0. If multiple images follow the edit equally, prioritize the one with the lowest number possible. 
Avoid selecting images that appear to follow the edit but are not edits of the original image. 
Additionally, avoid images where camera motion, zoom, or image quality differs significantly, or where the content does not appear stable relative to the original source. 
Respond with: “The selected edit is:x” where x is the number of your chosen edit.

"""
    
    content = []
    content.append({
        "type": "text",
        "text": sys,
    })
    for idx,img in enumerate(os.listdir(frame_dir)):
        encoded_image = encode_image(os.path.join(frame_dir,img))
        content.append({
            "type": "text",
            "text": f"here is image {idx}:",
        })
        content.append({
            "type": "image_url",
            "image_url":{"url": f"data:image/png;base64,{encoded_image}"}
        })
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]

    completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,

        )
    if hasattr(completion.choices[0], 'message'):
        # 获取完整消息内容
        full_message = completion.choices[0].message

        answer_content = full_message.content or ""
        print("choice", answer_content)
        
        # 使用正则表达式提取编辑编号
        pattern = r"The selected edit is:\s*(\d+)"
        match = re.search(pattern, answer_content)
        if match:
            selected_number = int(match.group(1))
            print(f"Selected edit number: {selected_number}")
            return selected_number
        else:
            print("Could not extract edit number from response")
            return None

def copy_selected_frames(root, meta_path, target_path):
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    
    with open(meta_path, "r") as f:  # 改为只读模式
        lines = f.readlines()
        
    for item in tqdm(os.listdir(root), desc="Processing video folders"):
        if item.startswith("video"):
            num = item.split("_")[1]
            meta = lines[int(num)-1]
            meta = json.loads(meta)
            prompt = meta["edit_prompt"]
            
            # 获取选择的帧
            choice = frame_selection(prompt, os.path.join(root, item))
            if choice is not None:
                # 构建源路径和目标路径
                source_frame_path = os.path.join(root, item, f"frame_{choice*4:04d}.png")
                
                # 根据data_id构建目标文件名
                data_id = meta.get("data_id", f"{num}")  # 如果没有data_id，则使用编号
                target_frame_path = os.path.join(target_path, f"output_{data_id}.png")
                
                # 复制文件
                if os.path.exists(source_frame_path):
                    shutil.copy2(source_frame_path, target_frame_path)
                    print(f"Copied: {source_frame_path} -> {target_frame_path}")
                else:
                    print(f"Warning: Source frame does not exist: {source_frame_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and copy frames based on AI evaluation.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing video folders")
    parser.add_argument("--meta_path", type=str, required=True, help="Path to meta JSONL file")
    parser.add_argument("--target_path", type=str, required=True, help="Target directory to copy selected frames to")
    
    args = parser.parse_args()
    
    copy_selected_frames(args.root, args.meta_path, args.target_path)