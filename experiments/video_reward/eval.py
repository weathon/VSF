from datasets import load_dataset
import random
import os
import shutil
import subprocess



ds = load_dataset("KwaiVGI/VideoGen-RewardBench")
random.seed(42)
prompts = random.choices(list(set(ds["eval"]["prompt"])), k=100)
os.makedirs("videos", exist_ok=True)

for i, prompt in enumerate(prompts):
    subprocess.run(['python3', 'wan_nag.py', '--prompt', prompt, '--video_id', str(i)])
    subprocess.run(['python3', 'wan_none.py', '--prompt', prompt, '--video_id', str(i)])
    subprocess.run(['python3', 'wan_vsf.py', '--prompt', prompt, '--video_id', str(i)])