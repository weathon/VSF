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
from concurrent.futures import ThreadPoolExecutor
import threading
import dotenv
dotenv.load_dotenv()

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
from openai import OpenAI

from google.colab import userdata
import base64
from PIL import Image
import io



client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=userdata.get('or'),
)


def generate(prompt, missing_element):
    completion = client.chat.completions.create(
      model="google/gemini-2.5-flash-image-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Generate an image of a house with no roof"
            },
          ]
        }
      ]
    )
    print(completion.choices[0].message.content)
    b64 = completion.choices[0].message.images[0]["image_url"]["url"]


    img_data = base64.b64decode(b64.split(",")[1].replace('\x00', ''))
    img = Image.open(io.BytesIO(img_data))

    return img

import tqdm

def process_prompt(prompt_data, seed):
    """Worker function to process a single prompt"""
    image = generate(prompt_data["prompt"], prompt_data["missing_element"])
    result = {"image": image, "prompt_data": prompt_data}
    
    if not args.eval_later:
        delta = judge.vqa(image, prompt_data["question_1"], prompt_data["question_2"])
        result["delta"] = delta
    
    return result

def run():
    wandb.init(project="nag-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    score_lock = threading.Lock()
    
    for seed in range(2):
        # Create tasks for thread pool
        tasks = [(prompt_data, seed) for prompt_data in dev_prompts]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            futures = [executor.submit(process_prompt, prompt_data, seed) for prompt_data, seed in tasks]
            
            # Process results as they complete
            for future in tqdm.tqdm(futures, desc=f"Seed {seed}"):
                result = future.result()
                image = result["image"]
                prompt_data = result["prompt_data"]
                
                if not args.eval_later:
                    delta = result["delta"]
                    with score_lock:
                        score += delta
                        total += 1
                    
                    from PIL import ImageDraw, ImageFont
                    wandb.log({"pos_score_overall":score[0]/total, "neg_score_overall":score[1]/total, "quality_score_overall": score[2]/total,"img": wandb.Image(image, caption=f"+: {prompt_data['prompt']}\n -: {prompt_data['missing_element']}"), 
                              "pos_score": delta[0], "neg_score": delta[1], "quality_score": delta[2]})
                else:
                    wandb.log({"img": wandb.Image(image, caption=f"{prompt_data['prompt']}****{prompt_data['missing_element']}")})

run()
