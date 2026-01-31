import http.client
import json
import base64
import os
import time


def generate_image_with_gemini(prompt, host="yinli.one", api_key="", output_filename="generated_image.png", aspect_ratio="1:1"):
    """
    使用Gemini API生成图像
    
    Args:
        prompt (str): 生成图像的提示文本
        host (str): API主机地址
        api_key (str): API密钥
        output_filename (str): 输出图像文件名
        aspect_ratio (str): 图像宽高比，默认为"1:1"
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    path = "/v1beta/models/gemini-3-pro-image-preview:generateContent"

    payload = {
        "contents": [
            {"parts":
                [
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": "1k"},
        },
    }

    conn = http.client.HTTPSConnection(host, timeout=300)
    try:
        conn.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        res = conn.getresponse()
        raw = res.read()

        if res.status != 200:
            print(f"API请求失败，状态码: {res.status}")
            return False

        try:
            obj = json.loads(raw.decode("utf-8"))
            
            # 取第一张图（如果有多张可改成循环）
            parts = obj["candidates"][0]["content"]["parts"]

            for p in parts:
                if "text" in p:
                    if "![image](" in p["text"]:
                        img_b64 = p["text"].replace("![image](", "").replace(")", "").split(',')[1]
                        with open(output_filename, "wb") as f:
                            f.write(base64.b64decode(img_b64))
                        print(f"图像已保存为 {output_filename}")
                        return True
                    else:
                        print("文本回复:", p["text"])
                if "inlineData" in p:
                    img_b64 = p["inlineData"]["data"]
                    with open(output_filename, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                    print(f"图像已保存为 {output_filename}")
                    return True
            else:
                print("没有找到 inlineData（没有图片返回）。")
                return False
                
        except json.JSONDecodeError:
            print("解析API响应失败")
            return False
            
    except Exception as e:
        print(f"请求过程中发生错误: {e}")
        return False
    finally:
        conn.close()


# 遍历文件夹处理jsonl文件
root = "/hsk/dataset_test/anti-physic"
for item in os.listdir(root):
    if os.path.isdir(os.path.join(root, item)) and item != "hunyuan":
        print("item:", item)
        json_data = os.path.join(root, item, f"{item}.jsonl")
        if os.path.exists(json_data):
            with open(json_data, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    print(f"input_prompt:", data["input_prompt"])
                    origin_scene_prompt = data["input_prompt"]
                    output_path = os.path.join(root, item, f"{item}_origin_{i}-gemini.png")
                    
                    # 检查图片是否已存在，若存在则跳过
                    if os.path.exists(output_path):
                        print(f"图片已存在，跳过生成: {output_path}")
                        continue
                    
                    # 生成图像
                    success = generate_image_with_gemini(
                        prompt=origin_scene_prompt,
                        output_filename=output_path
                    )
                    
                    if success:
                        print(f"图像已保存为 {output_path}")
                    else:
                        print(f"生成图像失败: 第{i}行")
        else:
            print(f"文件不存在: {json_data}")
