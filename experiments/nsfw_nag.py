import torch
import sys
from nag import NAGStableDiffusion3Pipeline
import json
import judge
import wandb
import numpy as np
import os
import argparse
from mod import moderate_image

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
        
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = NAGStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/nsfw_prompts.json", "r") as f:
    dev_prompts = json.load(f)

def run(nag_scale, nag_alpha, nag_tau):
    wandb.init(project="nag-sweep")
    score = 0
    total = 0
    seed = 42
    for i in dev_prompts:
        image = pipe(
            i["prompt"],
            nag_negative_prompt=i["missing_element"],
            guidance_scale=0.,
            nag_scale=nag_scale,
            nag_alpha=nag_alpha,
            nag_tau=nag_tau,
            num_inference_steps=8,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        score += moderate_image(image)
        total += 1
        wandb.log({"score": score / total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
        seed += 1
        
run(10, 0.5, 5)
# run(4, 0.125, 2.5)
