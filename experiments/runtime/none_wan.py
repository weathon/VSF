import sys
sys.path.append("../../")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video
import json
import argparse
with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import time
total_time = 0
count = 0
import wandb
import os
import numpy as np
wandb.init(project="compute", name="none_wan")
total_time = 0
count = 0


model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    adapter_name="lora"
) 
pipe = pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

for i in dev_prompts[:25]: 

    prompt = i["prompt"]
    neg_prompt = i["missing_element"]

    height = 480
    width = 832
    frames = 81

    pipe.set_adapters("lora", 0.5)

    start_time = time.time()
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=8,
        guidance_scale=0.0, 
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    end_time = time.time()
    total_time += (end_time - start_time)
    count += 1
    wandb.log({"time_per_image": total_time / count})
wandb.log({"peak_memory": torch.cuda.max_memory_allocated() / (1024 ** 3)})
