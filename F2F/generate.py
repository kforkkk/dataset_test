"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5B --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""
import os
import json
from .prompt_trans import encode_image, generate_video_description
import argparse
import logging
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

# Global variable to cache the model pipeline
_pipe_cache = {}

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def save_sample_frames(video_generate, output_folder, interval=4):
    """
    保存视频的采样帧到指定文件夹
    
    Args:
        video_generate: 生成的视频帧，可能是一个PIL图像列表或者numpy数组
        output_folder: 输出文件夹路径
        interval: 采样间隔，默认每4帧保存一次
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 确定帧的数量
    if isinstance(video_generate, list):
        total_frames = len(video_generate)
    else:
        total_frames = video_generate.shape[0] if len(video_generate.shape) >= 3 else 1
    
    # 提取并保存每interval帧
    saved_count = 0
    for i in range(0, total_frames, interval):
        if isinstance(video_generate, list):
            frame = video_generate[i]
        else:
            # 如果是numpy数组，转换为PIL图像
            frame_array = video_generate[i]
            if isinstance(frame_array, torch.Tensor):
                frame_array = frame_array.cpu().numpy()
            # 确保数值范围是正确的
            if frame_array.max() <= 1.0:
                frame_array = (frame_array * 255).clip(0, 255).astype(np.uint8)
            frame = Image.fromarray(frame_array)
        
        # 保存帧
        frame_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        frame.save(frame_path)
        saved_count += 1
        
    print(f"已保存 {saved_count} 帧到 {output_folder}")


def get_model_pipe(
    model_path: str,
    lora_path: str = None,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = "t2v"
):
    """
    获取模型管道，如果已存在缓存则直接返回，否则创建新的
    """
    global _pipe_cache
    
    # 创建缓存键
    cache_key = (model_path, lora_path, dtype, generate_type)
    
    if cache_key in _pipe_cache:
        print(f"使用缓存的模型管道: {cache_key}")
        return _pipe_cache[cache_key]
    
    print(f"初始化模型管道: {cache_key}")
    
    # 1. Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # 缓存模型管道
    _pipe_cache[cache_key] = pipe
    
    return pipe


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    pipe=None  # 新增参数，允许传入预加载的模型管道
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    # 如果没有传入模型管道，则获取一个
    if pipe is None:
        pipe = get_model_pipe(model_path, lora_path, dtype, generate_type)

    image = None
    video = None

    if generate_type == "i2v":
        image = load_image(image=image_or_video_path)
    elif generate_type == "v2v":
        video = load_video(image_or_video_path)

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = (480, 720)# RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    
    # export_to_video(video_generate, output_path, fps=fps)
    
    # # 创建输出文件夹名称
    # base_output_name = os.path.splitext(os.path.basename(output_path))[0]
    # sample_frames_folder = f"{base_output_name}_sampled_frames"
    #print(f"Saving sample frames to {sample_frames_folder}")
    
    # 保存每4帧的采样帧
    save_sample_frames(video_generate, output_path, interval=4)
    print("Video saved to {}".format(output_path))


def process_dataset_and_generate_videos(
    root_path: str,
    model_path: str,
    output_assets_dir: str = "./assets",
    generate_type: str = "i2v",
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: str = None  # 添加LoRA路径参数
):
    """
    处理数据集并生成视频
    
    Args:
        root_path: 包含input_data和meta.jsonl的根目录
        model_path: 模型路径
        output_assets_dir: 输出资产目录
        generate_type: 生成类型
        num_frames: 帧数
        width: 视频宽度
        height: 视频高度
        fps: 每秒帧数
        dtype: 数据类型
        lora_path: LoRA权重路径
    """
    # 创建输出目录
    os.makedirs(output_assets_dir, exist_ok=True)
    
    # 预加载模型管道
    pipe = get_model_pipe(model_path, lora_path, dtype, generate_type)
    
    # 定义路径
    image_data_dir = os.path.join(root_path, "input_data")
    meta_data_path = os.path.join(root_path, "meta.jsonl")
    enhanced_prompts_path = os.path.join(output_assets_dir, "aug_prompt.jsonl")
    
    # 读取之前已生成的增强提示词
    processed_entries = {}
    if os.path.exists(enhanced_prompts_path):
        with open(enhanced_prompts_path, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                data = json.loads(line.strip())
                # 使用editing_prompt和image_path作为唯一标识
                key = data.get("data_id","")
                processed_entries[key] = data
    
    # 读取原始meta数据
    meta_data_list = []
    with open(meta_data_path, "r", encoding="utf-8") as meta_file:
        for line in meta_file:
            data = json.loads(line.strip())
        
            editing_prompt = data.get("edit_prompt", "")  # 注意这里的键名可能有空格
            image_filename = f"data_{data.get("data_id", "")}.png"
            data_id = data.get("data_id", "")  # 使用原始数据中的data_id
            image_path = os.path.join(image_data_dir, image_filename)
            
            # 检查是否已经处理过
            entry_key = data_id
            if entry_key in processed_entries:
                tqdm.write(f"跳过已处理的条目: {editing_prompt}")
                continue
                
            meta_data_list.append({
                "data_id": data_id,
                "editing_prompt": editing_prompt,
                "image_filename": image_filename,
                "image_path": image_path
            })
    
    # 追加模式写入新增的条目
    with open(enhanced_prompts_path, "a", encoding="utf-8") as enhanced_file:
        # 先处理新的条目并写入文件，添加进度条
        for data in tqdm(meta_data_list, desc="Processing data entries"):
            editing_prompt = data["editing_prompt"]
            image_filename = data["image_filename"]
            image_path = data["image_path"]
            data_id = data["data_id"]  # 使用原始数据的data_id
            
            # 使用generate_video_description生成增强后的提示词
            print("Generating enhanced prompt for:", editing_prompt)
            enhanced_prompt = generate_video_description(editing_prompt, image_path)
            
            # 创建新的数据字典，包含增强后的提示词
            enhanced_data = {
                "data_id": data_id,  # 保存原始数据ID
                "editing_prompt": editing_prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_path
            }
            
            # 将增强后的提示词写入文件
            enhanced_file.write(json.dumps(enhanced_data, ensure_ascii=False) + "\n")
            
            # 为这个增强后的提示词生成视频，使用原始数据ID作为文件夹名
            video_filename = f"video_{data_id}.mp4"
            video_path = os.path.join(output_assets_dir, f"video_{data_id}")
            os.makedirs(video_path, exist_ok=True)  # 创建单独的视频文件夹
            
            # 构建完整视频路径
            #full_video_path = os.path.join(video_path, video_filename)
            
            # 生成视频
            generate_video(
                prompt=enhanced_prompt,
                model_path=model_path,
                image_or_video_path=image_path,  # 使用对应的图片路径
                output_path=video_path,
                generate_type=generate_type,
                num_frames=num_frames,
                width=width,
                height=height,
                fps=fps,
                dtype=dtype,
                pipe=pipe  # 传递模型管道
            )
    
    # 读取所有已存在的增强提示词
    all_enhanced_data = []
    if os.path.exists(enhanced_prompts_path):
        with open(enhanced_prompts_path, "r", encoding="utf-8") as enhanced_file:
            for line in enhanced_file:
                all_enhanced_data.append(json.loads(line.strip()))
    
    # 为所有已存在的增强提示词生成视频（如果视频文件不存在），添加进度条
    for data in tqdm(all_enhanced_data, desc="Generating videos for existing entries"):
        enhanced_prompt = data["enhanced_prompt"]
        image_path = data["image_path"]
        
        #image_path = os.path.join(image_data_dir, image_filename)
        
        # 使用数据中的data_id字段
        data_id = data.get("data_id", str(hash(enhanced_prompt) % 10000))  # 如果没有data_id则使用哈希值
        
        # 检查视频是否已经生成
        #video_filename = f"video_{data_id}.mp4"
        video_path = os.path.join(output_assets_dir, f"video_{data_id}")
        #full_video_path = os.path.join(video_path, video_filename)
        
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)  # 确保目录存在
            
            # 生成视频
            generate_video(
                prompt=enhanced_prompt,
                model_path=model_path,
                image_or_video_path=image_path,
                output_path=video_path,
                generate_type=generate_type,
                num_frames=num_frames,
                width=width,
                height=height,
                fps=fps,
                dtype=dtype,
                pipe=pipe  # 传递模型管道
            )
        else:
            tqdm.write(f"视频 {video_path} 已存在，跳过")

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     description="Generate a video from a text prompt using CogVideoX"
    # )
    # parser.add_argument(
    #     "--prompt", type=str, required=True, help="The description of the video to be generated"
    # )
    # parser.add_argument(
    #     "--image_or_video_path",
    #     type=str,
    #     default=None,
    #     help="The path of the image to be used as the background of the video",
    # )
    # parser.add_argument(
    #     "--model_path",
    #     type=str,
    #     default="THUDM/CogVideoX1.5-5B",
    #     help="Path of the pre-trained model use",
    # )
    # parser.add_argument(
    #     "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    # )
    # parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    # parser.add_argument(
    #     "--output_path", type=str, default="./output.mp4", help="The path save generated video"
    # )
    # parser.add_argument(
    #     "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance"
    # )
    # parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    # parser.add_argument(
    #     "--num_frames", type=int, default=81, help="Number of steps for the inference process"
    # )
    # parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    # parser.add_argument(
    #     "--height", type=int, default=None, help="The height of the generated video"
    # )
    # parser.add_argument(
    #     "--fps", type=int, default=16, help="The frames per second for the generated video"
    # )
    # parser.add_argument(
    #     "--num_videos_per_prompt",
    #     type=int,
    #     default=1,
    #     help="Number of videos to generate per prompt",
    # )
    # parser.add_argument(
    #     "--generate_type", type=str, default="t2v", help="The type of video generation"
    # )
    # parser.add_argument(
    #     "--dtype", type=str, default="bfloat16", help="The data type for computation"
    # )
    # parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    # args = parser.parse_args()
    # dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    process_dataset_and_generate_videos(
        root_path="/hsk/dataset_test/anti-physic/gemini_gen_antiphysic",
        model_path="/hsk/F2F/CogVideo/weight",
        output_assets_dir="/hsk/F2F/assets",
        dtype=torch.bfloat16,

    )

    # generate_video(
    #     prompt=args.prompt,
    #     model_path=args.model_path,
    #     lora_path=args.lora_path,
    #     lora_rank=args.lora_rank,
    #     output_path=args.output_path,
    #     num_frames=args.num_frames,
    #     width=args.width,
    #     height=args.height,
    #     image_or_video_path=args.image_or_video_path,
    #     num_inference_steps=args.num_inference_steps,
    #     guidance_scale=args.guidance_scale,
    #     num_videos_per_prompt=args.num_videos_per_prompt,
    #     dtype=dtype,
    #     generate_type=args.generate_type,
    #     seed=args.seed,
    #     fps=args.fps,
    # )


