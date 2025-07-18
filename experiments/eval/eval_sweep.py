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
from eval_run import evaluate_run
parser = argparse.ArgumentParser(description="Evaluate a sweep of runs.")
parser.add_argument("--sweep_id", type=str, required=True, help="The ID of the sweep to evaluate.")
args = parser.parse_args()
sweep_id = args.sweep_id

wandb_api = Api()

sweep = wandb_api.sweep(sweep_id)
eval_results = []
results_lock = threading.Lock()


# Run all runs in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(evaluate_run, run) for run in sweep.runs]
    
    # Wait for all threads to complete
    for future in futures:
        future.result()

# for run in sweep.runs:
#     evaluate_run(run)

# save results to a JSON file
with open(f"{sweep_id.replace('/', '_')}.json", 'w') as f:
    json.dump(eval_results, f, indent=4)
