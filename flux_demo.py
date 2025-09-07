import torch
from src.flux_pipeline import VSFFluxPipeline
import numpy as np
import imageio

pipe = VSFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

# prompt = "a canadian winter landscape in the style of a 19th century painting"
prompt = "a old style TV on the table with the screen taken off showing the inner components, detailed, high quality"
images = []
# for scale in np.arange(3.0, 9.0, 0.5):
image = pipe(
    prompt,
    negative_prompt="glass",
    guidance_scale=0.0,
    num_inference_steps=30, 
    max_sequence_length=256,
    offset=0.2,
    scale=3.8, 
    generator=torch.Generator("cpu").manual_seed(93)
).images[0].save(f"flux_demo.png")
# images.append(image)

# imageio.mimsave("flux_demo.mp4", images, fps=4)