from datasets import load_dataset
import random
import os
import shutil
import subprocess


ds = load_dataset("KwaiVGI/VideoGen-RewardBench")
random.seed(42)
prompts = list(ds["eval"]["prompt"])#random.choices(list(ds["eval"]["prompt"]), k=100)
random.shuffle(prompts)
os.makedirs("videos", exist_ok=True)

import wandb
wandb.init(project="wan14b-videos", name="wan14b-vsf")
for i, prompt in enumerate(prompts):
    if len(prompt) < 300:
        print("Fuck, prompt too short!")
        continue
    subprocess.run(['python3', 'wan14b.py', '--prompt', prompt, '--video_id', str(i)])
    wandb.log({"vsf": wandb.Video(f"videos/14b_vsf_{i:03d}.mp4", caption=prompt),
                "original": wandb.Video(f"videos/14b_original_{i:03d}.mp4", caption=prompt)})