
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

def run(scale):
    wandb.init(project="nasa-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    for seed in range(1): 
        for i in dev_prompts:
            image = pipe(
                i["prompt"],
                nag_negative_prompt=i["missing_element"],
                guidance_scale=0.,
                nag_scale=scale,
                num_inference_steps=8,
            ).images[0]

            if not args.eval_later:
                delta = judge.vqa(image, i["question_1"], i["question_2"])
                score += delta
                total += 1
                wandb.log({"pos_score_overall":score[0]/total, "neg_score_overall":score[1]/total, "quality_score_overall": score[2]/total,"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}"), 
                          "pos_score": delta[0], "neg_score": delta[1], "quality_score": delta[2]})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})


run(0.28)

