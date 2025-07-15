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

client = OpenAI()
class Score(BaseModel):
    image_positive_and_quality: float
    image_negative: float


def ask_gpt(image1: Image.Image, pos: str, neg: str) -> list[Score]:
    buf1 = io.BytesIO()
    image1 = image1.resize((448, 448))
    image1.save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    prompt = (
        f"You will get 1 image, you should rate it from 0-10 based on how well they follow the positive prompt and quality of the image ({pos}),"
        f"and how well they AVOID the negative prompt ({neg}), that means the more *unrelated* the negative prompt is to the image the higher score, only give 10 if the negative item is completely avoided without any artifacts, "
        f"For each item you can rate from 0.0-10.0, 0 means bad and 10 means good. Your score should be fine grained to 1"
        f"Important: When the negative prompt is contradicted with positive prompt or quality following the negative prompt should not be a reason to decrease score for the positive and quality score. (such as negative prompt being 'car' while positive being 'New York street', showing no cars should not be a reason to decrease score for positive, even though it does not look like a New York street anymore. Similar goes for removing an important part of an item, like a car but with negative prompt of wheels, removing wheels making it less of a car, however, when you rate it, you should NOT decrease the score based on this) "
    ) 

    completion = client.beta.chat.completions.parse(
        model="o4-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
            ]},
        ],
        response_format=Score,
        reasoning_effort="low"
    )

    answer_first = completion.choices[0].message.parsed


    answer = np.array((answer_first.image_positive_and_quality, answer_first.image_negative))
    return answer
