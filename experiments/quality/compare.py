import torch
import sys
sys.path.append("../..")
sys.path.append("../")
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
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")


image = pipe(
    "portrait of barbaric spanish conquistador, symmetrical, by yoichi hatakenaka, studio ghibli and dan mumford",
    negative_prompt="low quality, blurry, low resolution, poor quality, unclear",
    guidance_scale=0.,
    scale=1,
    offset=0.3,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(3),
).images[0]
image.save("vsf.png")

image = pipe(
    "portrait of barbaric spanish conquistador, symmetrical, by yoichi hatakenaka, studio ghibli and dan mumford",
    negative_prompt="",
    guidance_scale=0.,
    scale=0,
    offset=1000,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(3),
).images[0]
image.save("original.png")