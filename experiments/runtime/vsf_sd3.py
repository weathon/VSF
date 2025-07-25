import torch
import sys
sys.path.append("..")
from src.sd3_pipeline import VSFStableDiffusion3Pipeline
import json
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

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import time
total_time = 0
count = 0
torch.reset_peak_memory_stats()

wandb.init(project="compute", name="vsf_sd3")
for i in dev_prompts[:25]:
    time_start = time.time()
    image = pipe(
        i["prompt"],
        negative_prompt=i["missing_element"],
        guidance_scale=0.,
        scale=3.8,
        offset=0.1,
        num_inference_steps=8,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]
    time_end = time.time()
    total_time += (time_end - time_start)
    count += 1
    wandb.log({"time_per_image": total_time / count})

wandb.log({"peak_memory": torch.cuda.max_memory_allocated() / (1024 ** 3)})
