import torch
from src.flux_pipeline import VSFFluxPipeline

pipe = VSFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

prompt = "a canadian winter landscape in the style of a 19th century painting"
image = pipe(
    prompt,
    negative_prompt="snow",
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    scale=6,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-schnell.png")
