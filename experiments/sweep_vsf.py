import torch
import sys
sys.path.append("..")
from src.sd3_pipeline import VSFStableDiffusion3Pipeline
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
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)

seed = 1999
def run():
    wandb.init(project="vsf-sweep")
    scale = wandb.config.scale
    offset = wandb.config.offset
    scores = np.zeros(2)
    os.makedirs(f"results_vsf/{wandb.run.id}")
    total = 0
    for idx, i in enumerate(dev_prompts):
        image = pipe(
            i["prompt"],
            negative_prompt=i["missing_element"],
            guidance_scale=0.,
            scale=scale,
            offset=offset,
            num_inference_steps=8,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        if not args.eval_later:
            scores += judge.ask_gpt(image, i["prompt"], i["missing_element"])
            wandb.log({"pos_score": scores[0]/total, "neg_score": scores[1]/total, "total_score": (scores[0] * 0.4 + scores[1] * 0.6)/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            total += 1
        else:
            # wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            image.save(f"results_vsf/{wandb.run.id}/{idx:03d}.png")
            
sweep_configuration = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "total_score"},
    "parameters": {
        "scale": {"min": 0.0, "max": 6.0},
        "offset": {"min": 0.0, "max": 0.5}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="vsf-sweep")

wandb.agent(sweep_id, function=run, count=32)
