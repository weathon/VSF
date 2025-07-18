import sys
import os
import numpy as np
sys.path.append("../")

from src.sd3_pipeline import VSFStableDiffusion3Pipeline
import torch
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
    torch_dtype=torch.bfloat16,
).to("cuda:2")
os.makedirs("pngs", exist_ok=True)
for scale in range(0,6):
    for bias in np.arange(0, 0.6, 0.1):
        prompt = "a cat making a cake in the kitchen, the cat is wearing a chef's hat and apron, the kitchen is modern and well-lit"
        image = pipe(
            prompt,
            negative_prompt="chef hat, apron",
            guidance_scale=0.0,
            num_inference_steps=8,
            max_sequence_length=256,
            scale=scale,
            offset=bias,
            generator=torch.Generator("cpu").manual_seed(19)
        ).images[0]
        image.save(f"pngs/demo_{scale}_{bias}.png")