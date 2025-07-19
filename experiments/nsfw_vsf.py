import torch
import sys
sys.path.append("..")
from src.sd3_pipeline import VSFStableDiffusion3Pipeline
import json
import judge
import wandb
import numpy as np
from mod import moderate_image
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

with open("../prompts/nsfw_prompts.json", "r") as f:
    dev_prompts = json.load(f)

seed = 42
def run(scale, offset):
    wandb.init(project="vsf-sweep")
    score = 0
    total = 0
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
        score += moderate_image(image)
        total += 1
        wandb.log({"score": score / total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
        seed += 1 
run(4.5, 0.2)