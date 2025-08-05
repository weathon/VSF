import sys
sys.path.append("..")
sys.path.append("../..")
import sys
import json
import judge
import wandb
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
        


with open("../../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)


import torch
import io
from openai import OpenAI
import base64   
from PIL import Image
client = OpenAI()
def generate(prompt, missing_element):
    prompt= prompt + ", but with no " + missing_element
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
    )

    image_base64 = result.data[0].b64_json
    buffer = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(buffer)
    return image

import tqdm
def run():
    wandb.init(project="nag-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    for seed in range(2):
        for i in tqdm.tqdm(dev_prompts):
            # pipe = pipe.to("cuda")
            image = generate(i["prompt"], i["missing_element"])
            if not args.eval_later:
                delta = judge.vqa(image, i["question_1"], i["question_2"])
                score += delta
                total += 1
                from PIL import ImageDraw, ImageFont
                wandb.log({"pos_score_overall":score[0]/total, "neg_score_overall":score[1]/total, "quality_score_overall": score[2]/total,"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}"), 
                          "pos_score": delta[0], "neg_score": delta[1], "quality_score": delta[2]})
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})

run()
