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

def moderate_image(image: Image.Image) -> bool:
    
    buf1 = io.BytesIO()
    image.save(buf1, format="PNG")
    b64 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",                    
                }
            },
        ],
    )
    print(response)
    return response.results[0].category_scores.sexual