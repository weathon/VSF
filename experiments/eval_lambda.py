import os
import json
import base64
import io
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import random
import dotenv
from concurrent.futures import ThreadPoolExecutor

dotenv.load_dotenv()
from openai import OpenAI

openai_api_key = os.environ["key"]
openai_api_base = "https://api.lambda.ai/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class Ans(BaseModel):
    reasoning: str
    answer_1: bool
    answer_2: bool
    quality: float

def vqa(image1: Image.Image, question1: str, question2: str) -> np.ndarray:
    buf1 = io.BytesIO()
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    prompt = f"Answer the following questions, only answer with boolean, only answer True if it follow all the conditions, otherwise answer False. Answer only with True or False. For first question, you should answer if the main object is there, no matter if a key element described in second question is missing from it. For second question asking if something is missing, answer True if it is missing or invisible, otherwise False. That means, if you cannot see it, label it as missing, do not presume from other elements. Give a reasoning process before you response. Also, rate the quality of the image from 0-1, if the image is missing the key element mentioned in the second question, do NOT use that as a reason to decrease the quality score. Decrease the quality score if you see artifacts, blurring, unnatural color, noise, deformed objects, double vision, etc. If the image is very blurry and hard to see the main object, give it a low quality score."

    completion = client.beta.chat.completions.parse(
        model="llama-4-maverick-17b-128e-instruct-fp8",
        messages=[ 
            {"role": "system", "content": prompt}, 
            {"role": "user", "content": [
                {"type": "text", "text": f"Question 1 is: {question1}, Question 2 is: {question2}."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
            ]},
        ], 
        response_format=Ans,
    )

    answer = completion.choices[0].message.parsed
    return [answer.answer_1, answer.answer_2, answer.quality], answer.reasoning
    
import wandb
from PIL import ImageDraw, ImageFont
wandb.init()
with open("../prompts/test_prompts.json.new", "r") as f:
    prompts = json.load(f)

def process_image(args):
    idx, image, run, prompts = args
    question1 = prompts[int(image.replace(".png",""))]["question_1"]
    question2 = prompts[int(image.replace(".png",""))]["question_2"]
    image_path = f"results_vsf/{run}/{image}"
    image1 = Image.open(image_path).convert("RGB")
    answer, reasoning = vqa(image1, question1, question2.replace("missing", "missing or not visible"))
    return int(image.replace(".png","")), answer, reasoning
import tqdm
for run in tqdm.tqdm(os.listdir("results_vsf")):
    images = os.listdir(f"results_vsf/{run}")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, image, run, prompts) for idx, image in enumerate(images) if image.endswith(".png")]
    
    # Process images in parallel with max 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_image, args_list))
    
    # Sort results by index to maintain original order
    results.sort(key=lambda x: x[0])
    data = [{"ans": result[1], "reasoning": result[2]} for result in results]
    
    with open("results_vsf/" + run + "/vqa.json", "w") as f:
        json.dump(data, f)
    print("Finished processing run: ", run)


