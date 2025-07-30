import torch
from pipeline import NASAStableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = NASAStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "a bike by the wall, during sunset, in the style of a 1980s photograph, high quality, detailed, realistic"
nag_negative_prompt = "wheels"

image = pipe(
    prompt,
    nag_negative_prompt=nag_negative_prompt,
    guidance_scale=0.,
    nag_scale=0.25,
    num_inference_steps=8,
).images[0].save("output.png")