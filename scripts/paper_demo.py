from datasets import load_dataset
import random
import os
import shutil
import subprocess

prompts = [
    "A lava river flowing through a volcanic landscape, dark rocky terrain. The camera captures the the flow of lava. The sky is dark with ash clouds.",
    "A plane flying over a snowy mountain range, with the sun setting in the background. The camera captures the plane's silhouette against the colorful sky and the snow-covered peaks below.",
    "A machine learning scientist working in a lab, analyzing data on a computer screen. The camera captures the scientist's focused expression and the complex algorithms displayed on the screen.",
]
neg_prompts = [
    "plane wings",
    "male with glasses"
    "red hot, bright, glow",
]

import wandb
wandb.init(project="wan14b-videos", name="wan14b-vsf")
for i, prompt in enumerate(prompts):
    subprocess.run(['python3', 'wan_vsf.py', '--prompt', prompts[i], '--neg_prompt', neg_prompts[i], '--video_id', str(i)])
    subprocess.run(['python3', 'wan_nag.py', '--prompt', prompts[i], '--neg_prompt', neg_prompts[i], '--video_id', str(i)])
    subprocess.run(['python3', 'wan_none.py', '--prompt', prompts[i], '--neg_prompt', neg_prompts[i], '--video_id', str(i)])