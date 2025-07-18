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

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)

def run(nag_scale, nag_alpha, nag_tau):
    wandb.init(project="nag-sweep")
    score = np.array([0, 0], dtype=int)
    total = 0
    for seed in range(1):
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
                delta = judge.vqa(image, i["question_1"], i["question_2"])
                score += delta
                total += 1
                # show the score as text on the image using PIL
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype("DejaVuSans.ttf", 50)
                text = f"{delta}, -: {i['missing_element']}"
                draw.text((10, 10), text, fill="white", font=font)
                text = f"{delta[0]}, {delta[1]}, -: {i['missing_element']}"
                draw.text((10, 10), text, fill="white")
                wandb.log({"pos_score":score[0]/total, "neg_score":score[1]/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})

run(10, 0.5, 5)
# run(4, 0.125, 2.5)
