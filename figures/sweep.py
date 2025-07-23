import sys
import os
import numpy as np
sys.path.append("../")

from src.sd3_pipeline import VSFStableDiffusion3Pipeline
import torch
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
    torch_dtype=torch.bfloat16,
).to("cuda")
os.makedirs("pngs", exist_ok=True)
for i, scale in enumerate(np.arange(0.5, 3, 0.5)):
    for j, bias in enumerate(np.arange(0, 0.6, 0.2)):
        prompt = "a cat making a cake in the kitchen, the cat is wearing a chef's apron, the kitchen is modern and well-lit. The cake is colorful and has a cherry on top, the cat is smiling and looks happy, the kitchen has a large window with sunlight coming in."
        image = pipe(
            prompt,
            negative_prompt="chef hat",
            guidance_scale=0.0,
            num_inference_steps=8,
            max_sequence_length=256,
            scale=scale,
            offset=bias,
            generator=torch.Generator("cpu").manual_seed(19)
        ).images[0]
        image.save(f"pngs/demo_{i}_{j}.png")