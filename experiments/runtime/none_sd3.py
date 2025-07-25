import torch
import sys
from diffusers import StableDiffusion3Pipeline
import json
import judge
import wandb
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
        
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import time
total_time = 0
count = 0
wandb.init(project="compute", name="none_sd3")
def run(nag_scale, nag_alpha, nag_tau):
    score = np.array([0, 0], dtype=int)
    total = 0
    for i in dev_prompts[:25]:
        time_start = time.time()
        image = pipe(
            i["prompt"],
            guidance_scale=0.,
            nag_scale=nag_scale,
            nag_alpha=nag_alpha,
            nag_tau=nag_tau,
            num_inference_steps=8,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        time_end = time.time()
        total_time += (time_end - time_start)
        count += 1
        wandb.log({"time_per_image": total_time / count})
wandb.log({"peak_memory": torch.cuda.max_memory_allocated() / (1024 ** 3)})

run(10, 0.5, 5)
