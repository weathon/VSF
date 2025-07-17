import torch
from src.flux_pipeline import VSFFluxPipeline
import numpy as np
import imageio

pipe = VSFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

prompt = "a canadian winter landscape in the style of a 19th century painting"
images = []
for scale in np.arange(1, 9, 0.1):
    image = pipe(
        prompt,
        negative_prompt="snow on the ground",
        guidance_scale=0.0,
        num_inference_steps=8,
        max_sequence_length=256,
        scale=scale,
        generator=torch.Generator("cpu").manual_seed(19)
    ).images[0]
    images.append(image)

imageio.mimsave("flux_demo.mp4", images, fps=4)