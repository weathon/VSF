import openai
from wandb.apis.public import Api
from PIL import Image
from io import BytesIO
import requests
import sys
sys.path.append("../")
import judge
import tqdm
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Evaluate a run of runs.")
parser.add_argument("--run_id", type=str, required=True, help="The ID of the run to evaluate.")
args = parser.parse_args()

wandb_api = Api()

def evaluate_run(run):
    config = run.config
    pos_score = 0
    neg_score = 0
    count = 0
    
    print(f"Starting evaluation for run {run.id}")
    
    for i, row in tqdm.tqdm(list(run.history().iterrows()), desc=f"Run {run.id}"):
        img_path = row["img"]["path"]
        base_url = f"https://api.wandb.ai/files/{run.entity}/{run.project}/{run.id}/{img_path}"
        response = requests.get(base_url, stream=True)
        img = Image.open(BytesIO(response.content))
        pos = row["img"]["caption"].split("\n -: ")[0].replace("+: ", "")
        neg = row["img"]["caption"].split("\n -: ")[1]
        score_1 = judge.ask_gpt(
            img,
            pos=pos,
            neg=neg
        )
        score_2 = judge.ask_gpt(
            img,
            pos=pos,
            neg=neg
        )
        score = (score_1 + score_2) / 2  # Average the two samples
        pos_score += score[0]
        neg_score += score[1]
        count += 1
    
    result = {
        "run_id": run.id,
        "config": config,
        "pos_score": pos_score/count,
        "neg_score": neg_score/count,
        "total_score": (pos_score + neg_score) / (2 * count) 
    }

    
    print(f"Completed evaluation for run {run.id}")
    return result

res = evaluate_run(wandb_api.run(args.run_id))
with open(f"{args.run_id.replace('/', '_')}_eval.json", 'w') as f:
    json.dump(res, f, indent=4)
    print(f"Results saved to {args.run_id.replace('/', '_')}_eval.json")