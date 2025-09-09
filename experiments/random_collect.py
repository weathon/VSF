# import torch
# import sys
# sys.path.append("..")
# # from src.sd3_pipeline import VSFStableDiffusion3Pipeline
# from nag import NAGStableDiffusion3Pipeline

# import json
# import judge
# import wandb
# import numpy as np
# import dotenv
# dotenv.load_dotenv()
# import argparse

# parser = argparse.ArgumentParser(description="Run NAG sweep")
# parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
# args = parser.parse_args()

# model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
# pipe = NAGStableDiffusion3Pipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
# )
# pipe.to("cuda")

# with open("../prompts/test_prompts.json.new", "r") as f:
#     dev_prompts = json.load(f)
# import random
# import os
# import json
# os.makedirs("random_collect", exist_ok=True)
# def run():
#     data = []
#     score = np.array([0, 0], dtype=int)
#     total = 0
#     for seed in range(1):
#         for i in dev_prompts[:30]:
#             image = pipe(
#                 i["prompt"],
#                 negative_prompt=i["missing_element"],
#                 guidance_scale=0.,
#                 nag_scale=random.uniform(1, 24),
#                 nag_alpha=random.uniform(0.0, 1.0), 
#                 nag_tau=random.uniform(1, 10),
#                 num_inference_steps=8,
#                 generator=torch.Generator("cuda").manual_seed(seed),
#             ).images[0]
#             scores = judge.vqa(image, i["question_1"], i["question_2"])
#             filename = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
#             image.save(f"random_collect/{filename}.png")
#             data.append({
#                 "filename": filename,
#                 "scores": list(scores),
#             })
#     with open("random_collect/data2.json", "w") as f:
#         json.dump(data, f)
# run()

# import torch
# import sys
# sys.path.append("..")
# from src.sd3_pipeline import VSFStableDiffusion3Pipeline
# import json
# import judge
# import wandb
# import numpy as np
# import dotenv
# dotenv.load_dotenv()
# import argparse

# parser = argparse.ArgumentParser(description="Run NAG sweep")
# parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
# args = parser.parse_args()

# model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
# pipe = VSFStableDiffusion3Pipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
# )
# pipe.to("cuda")

# with open("../prompts/test_prompts.json.new", "r") as f:
#     dev_prompts = json.load(f)
# import random
# import os
# import json
# os.makedirs("random_collect", exist_ok=True)
# def run():
#     data = []
#     score = np.array([0, 0], dtype=int)
#     total = 0
#     for seed in range(1):
#         for i in dev_prompts[:30]:
#             image = pipe(
#                 i["prompt"],
#                 negative_prompt=i["missing_element"],
#                 guidance_scale=0.,
#                 scale=random.uniform(0, 8),
#                 offset=random.uniform(-0.1, 0),
#                 num_inference_steps=8,
#                 generator=torch.Generator("cuda").manual_seed(seed),
#             ).images[0]
#             scores = judge.vqa(image, i["question_1"], i["question_2"])
#             filename = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
#             image.save(f"random_collect/{filename}.png")
#             data.append({
#                 "filename": filename,
#                 "scores": list(scores),
#             })
#     with open("random_collect/data.json", "w") as f:
#         json.dump(data, f)
# run()




import torch
import sys
sys.path.append("..")
from nasa.pipeline import NASAStableDiffusion3Pipeline
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
pipe = NASAStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import random
import os
import json
os.makedirs("random_collect", exist_ok=True)
def run():
    data = []
    score = np.array([0, 0], dtype=int)
    total = 0
    for seed in range(1):
        for i in dev_prompts[:30]:
            image = pipe(
                i["prompt"],
                negative_prompt=i["missing_element"],
                guidance_scale=0.,
                scale=random.uniform(0, 0.5),
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            scores = judge.vqa(image, i["question_1"], i["question_2"])
            filename = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
            image.save(f"random_collect/{filename}.png")
            data.append({
                "filename": filename,
                "scores": list(scores),
            })
    with open("random_collect/data.json", "w") as f:
        json.dump(data, f)
run()
