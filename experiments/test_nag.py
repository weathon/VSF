import torch
import sys
from nag import NAGStableDiffusion3Pipeline
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
pipe = NAGStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/test_prompts.json", "r") as f:
    dev_prompts = json.load(f)

def run(nag_scale, nag_alpha, nag_tau):
    wandb.init(project="nag-sweep")
    scores = np.zeros(2)
    total = 0
    for seed in range(2):
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
            if not args.eval_later:
                scores += judge.ask_gpt(image, i["prompt"], i["missing_element"])
                total += 1
                # show the score as text on the image using PIL
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype("DejaVuSans.ttf", 50)
                text = f"Pos: {scores[0]/total:.4f}, Neg: {scores[1]/total:.4f}, -: {i['missing_element']}"
                draw.text((10, 10), text, fill="white", font=font)
                wandb.log({"pos_score": scores[0]/total, "neg_score": scores[1]/total, "total_score": (scores[0] * 0.4 + scores[1] * 0.6)/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})

run(5.9356, 0.28685, 6.9511)