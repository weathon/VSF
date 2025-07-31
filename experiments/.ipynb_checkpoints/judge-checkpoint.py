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

dotenv.load_dotenv()

openai_api_key = os.environ["key"]
openai_api_base = "https://api.lambda.ai/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# client = OpenAI()
# client = OpenAI(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )
class Score(BaseModel):
    image_positive_and_quality: int
    image_negative: int



def ask_gpt(image1: Image.Image, pos: str, neg: str) -> list[Score]:
    buf1 = io.BytesIO()
    image1 = image1.resize((448, 448))
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    prompt = (
         f"Answer the following questions, only answer with boolean, only answer True if it follow all the conditions, otherwise answer False. Answer only with True or False. For first question, you should answer if the main object is there, no matter if a key element described in second question is missing from it. For second question asking if something is missing, answer True if it is missing or invisible, otherwise False. That means, if you cannot see it, label it as missing, do not presume from other elements. Give a reasoning process before you response. Also, rate the quality of the image from 0-1, if the image is missing the key element mentioned in the second question, do NOT use that as a reason to decrease the quality score. Decrease the quality score if you see artifacts, blurring, unnatural color, noise, deformed objects, double vision, etc. If the image is very blurry and hard to see the main object, give it a low quality score."
    ) 

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[ 
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
            ]},
        ],
        response_format=Score,
        store=True
    )

    answer_first = completion.choices[0].message.parsed


    answer = np.array((answer_first.image_positive_and_quality, answer_first.image_negative))
    return answer


class Ans(BaseModel):
    reasoning: str
    answer_1: bool
    answer_2: bool
    quality: float

def vqa(image1: Image.Image, question1: str, question2: str) -> np.ndarray:
    buf1 = io.BytesIO()
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    prompt = f"Answer the following questions, only answer with boolean, only answer True if it follow all the conditions, otherwise answer False. Answer only with True or False. For first question, you should answer if the main object is there, no matter if a key element described in second question is missing from it. For second question asking if something is missing, answer True if it is missing or invisible, otherwise False. That means, if you cannot see it, label it as missing, do not presume from other elements. Give a reasoning process before you response. Also, rate the quality of the image from 0-1, if the image is missing the key element mentioned in the second question, do NOT use that as a reason to decrease the quality score. "

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
    return np.array([answer.answer_1, answer.answer_2, answer.quality], dtype=float)
