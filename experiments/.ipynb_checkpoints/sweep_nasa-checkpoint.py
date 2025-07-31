
import torch
import sys
sys.path.append("..")
from src.sd3_pipeline import VSFStableDiffusion3Pipeline
import json
import judge
import wandb
import numpy as np
import dotenv
dotenv.load_dotenv()
import argparse

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
from nasa.pipeline import NASAStableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = NASAStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import os
def run():
    wandb.init(project="nasa-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    scale = wandb.config.scale
    seed = 1999
    os.makedirs(f"results_nasa/{wandb.run.id}")
    for idx, i in enumerate(dev_prompts):
        image = pipe(
            i["prompt"],
            nag_negative_prompt=i["missing_element"],
            guidance_scale=0.,
            nag_scale=scale,
            num_inference_steps=8,
        ).images[0]

        if not args.eval_later:
            scores += judge.ask_gpt(image, i["prompt"], i["missing_element"])
            wandb.log({"pos_score": scores[0]/total, "neg_score": scores[1]/total, "total_score": (scores[0] * 0.4 + scores[1] * 0.6)/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            total += 1
        else:
            # wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            image.save(f"results_nasa/{wandb.run.id}/{idx:03d}.png")
            
sweep_configuration = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "total_score"},
    "parameters": {
        "scale": {"min": 0.0, "max": 0.5},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="nasa-sweep")

wandb.agent(sweep_id, function=run, count=66) 
