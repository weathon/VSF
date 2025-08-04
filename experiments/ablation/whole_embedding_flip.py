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
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
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
    os.makedirs(f"results_wef/{wandb.run.id}")
    total = 0
    for idx, i in enumerate(dev_prompts):
        prompt = dev_prompts[idx]["prompt"]
        negative_prompt = dev_prompts[idx]["missing_element"]
        (
            pos_prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            do_classifier_free_guidance=False,
        )

        (
            neg_prompt_embeds,
            _,
            neg_pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt,
            prompt_3=negative_prompt,
            do_classifier_free_guidance=False,
        )

        image = pipe(
            prompt_embeds=torch.cat([pos_prompt_embeds, -scale * neg_prompt_embeds], dim=1),
            pooled_prompt_embeds=pooled_prompt_embeds, 
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=8,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]

        if not args.eval_later:
            scores += judge.ask_gpt(image, i["prompt"], i["missing_element"])
            wandb.log({"pos_score": scores[0]/total, "neg_score": scores[1]/total, "total_score": (scores[0] * 0.4 + scores[1] * 0.6)/total, "img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            total += 1
        else:
            # wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})
            image.save(f"results_wef/{wandb.run.id}/{idx:03d}.png")
            
import numpy as np
sweep_configuration = {
    "method": "grid", 
    "metric": {"goal": "maximize", "name": "total_score"},
    "parameters": {
        "scale": {"values": list(np.arange(0.1, 6, 0.3))},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="vsf-sweep")

wandb.agent(sweep_id, function=run, count=32)
