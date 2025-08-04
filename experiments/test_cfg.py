import torch
import sys
sys.path.append("../..")
sys.path.append("../")
import torch
from diffusers import StableDiffusion3Pipeline


import json
import os

import wandb
import numpy as np
import dotenv
dotenv.load_dotenv()
import argparse

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
if not args.eval_later:
    import judge
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.to("cuda")

with open("../../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)

seed = 1999
def run():
    wandb.init(project="vsf-sweep")
    scale = wandb.config.scale
    scores = np.zeros(2)
    os.makedirs(f"results_cfg/{wandb.run.id}")
    total = 0
    for seed in range(2):
        for idx, i in enumerate(dev_prompts[:100]):
            prompt = dev_prompts[idx]["prompt"]
            negative_prompt = dev_prompts[idx]["missing_element"]
    
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=scale,
                num_inference_steps=28,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

            if not args.eval_later:
                delta = judge.vqa(image, i["question_1"], i["question_2"])
                score += delta
                total += 1
                # show the score as text on the image using PIL
                from PIL import ImageDraw, ImageFont
                wandb.log({"pos_score":score[0]/total, "neg_score":score[1]/total, "quality_score": score[2]/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})



# 3: Start the sweep
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="vsf-sweep")

# wandb.agent(sweep_id, function=run)
fun(10)