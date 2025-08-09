import sys
sys.path.append("..")
sys.path.append("../..")
import sys
import json
import judge
import wandb
import numpy as np
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
        


with open("../../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)


import torch
import io
from openai import OpenAI
import base64   
from PIL import Image
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model

model_name = "Qwen/Qwen-Image"

with no_init_weights():
    transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
            model_name, subfolder="transformer",
        ),
    ).to(torch.bfloat16)

DFloat11Model.from_pretrained(
    "DFloat11/Qwen-Image-DF11",
    device="cpu",
    cpu_offload=False,
    bfloat16_model=transformer,
)

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

pipe = pipe.to("cuda")
def generate(prompt, missing_element):
    width, height = 1328, 1328
    image = pipe(
        prompt=f"{prompt}; Ultra HD, 4K, cinematic composition.",
        negative_prompt=missing_element,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=7,
        generator=torch.Generator(device="cuda").manual_seed(0)
    ).images[0]

    return image

import tqdm

def process_prompt(prompt_data, seed):
    """Worker function to process a single prompt"""
    image = generate(prompt_data["prompt"], prompt_data["missing_element"])
    result = {"image": image, "prompt_data": prompt_data}
    
    if not args.eval_later:
        delta = judge.vqa(image, prompt_data["question_1"], prompt_data["question_2"])
        result["delta"] = delta
    
    return result

def run():
    wandb.init(project="nag-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    score_lock = threading.Lock()
    
    for seed in range(2):
        # Create tasks for thread pool
        tasks = [(prompt_data, seed) for prompt_data in dev_prompts]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            futures = [executor.submit(process_prompt, prompt_data, seed) for prompt_data, seed in tasks]
            
            # Process results as they complete
            for future in tqdm.tqdm(futures, desc=f"Seed {seed}"):
                result = future.result()
                image = result["image"]
                prompt_data = result["prompt_data"]
                
                if not args.eval_later:
                    delta = result["delta"]
                    with score_lock:
                        score += delta
                        total += 1
                    
                    from PIL import ImageDraw, ImageFont
                    wandb.log({"pos_score_overall":score[0]/total, "neg_score_overall":score[1]/total, "quality_score_overall": score[2]/total,"img": wandb.Image(image, caption=f"+: {prompt_data['prompt']}\n -: {prompt_data['missing_element']}"), 
                              "pos_score": delta[0], "neg_score": delta[1], "quality_score": delta[2]})
                else:
                    wandb.log({"img": wandb.Image(image, caption=f"+: {prompt_data['prompt']}\n -: {prompt_data['missing_element']}")})

run()
