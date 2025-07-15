import torch
import sys
sys.path.append("..")
from src.pipeline import VSFStableDiffusion3Pipeline
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
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/dev_prompts.json", "r") as f:
    dev_prompts = json.load(f)

def run():
    wandb.init(project="vsf-sweep")
    scale = wandb.config.scale
    offset = wandb.config.offset
    scores = np.zeros(2)
    total = 0
    for seed in range(2):
        for i in dev_prompts:
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
                total += 1
                wandb.log({"pos_score": scores[0]/total, "neg_score": scores[1]/total, "total_score": (scores[0] * 0.4 + scores[1] * 0.6)/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
        
sweep_configuration = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "total_score"},
    "parameters": {
        "scale": {"min": 0.0, "max": 5.0},
        "offset": {"min": 0.0, "max": 0.5}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="vsf-sweep")

wandb.agent(sweep_id, function=run, count=16)