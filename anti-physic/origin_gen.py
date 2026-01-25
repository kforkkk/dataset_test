from transformers import AutoModelForCausalLM
import os
import json

# Load the model
model_id = "/hsk/HunyuanImage-3.0/weight"
# Currently we can not load the model using HF model_id `tencent/HunyuanImage-3.0` directly 
# due to the dot in the name.

kwargs = dict(
    attn_implementation="flash_attention_2",     # Use "flash_attention_2" if FlashAttention is installed
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="flashinfer",   # Use "flashinfer" if FlashInfer is installed
)
# while True:
#     try:
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)
print("model loading done")
    #     break
    # except:
    #     print("model loading fail")

# # generate the image
# prompt = "A brown and white dog is running on the grass"
# image = model.generate_image(prompt=prompt, stream=True)
# image.save("image.png")


import time

def generate_image(prompt, max_retries=100000):
    for i in range(max_retries + 1):
        try:
            image = model.generate_image(prompt=prompt, stream=True)
            print("success")
            return image
        except Exception as e:
            print("error:", e)
            if i < max_retries:
                print("retrying...")
                time.sleep(1)  # 等待1秒再试
            else:
                print("Failed after", max_retries, "retries.")
                raise  # 或 return None，根据需求

root = "/hsk/dataset_test/anti-physic"
for item in os.listdir(root):
    if os.path.isdir(os.path.join(root, item)):
        print("item:",item)
        json_data = os.path.join(root, item, f"{item}.jsonl")
        with open(json_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                data = json.loads(line)
                print(data)
                origin_scene_prompt = data["formal_reasonable_scene"]
                origin_image = generate_image(origin_scene_prompt)
                origin_image.save(os.path.join(root, item, f"{item}_origin_{i}.png"))
    