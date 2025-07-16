# VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip

Paper and benchmark coming soon
## Introduction
This project introduces a new method called Value Sign Flip (VSF) that improves how image generation models handle negative prompts.

Problem: Modern few-step text-to-image models often struggle to properly exclude concepts described in negative prompts. Existing methods (CFG) either don’t work well or require heavy changes to the model (NegationCLIP).

Solution (VSF): We propose a lightweight technique that flips the value vector of negative prompt embeddings during attention. This cancels out unwanted features without retraining or needing access to classifier-free guidance.

Key Advantages:

- ⚡ Works with few-step and even single-step generation models (currently only supports SD3.5).
- 🔧 Requires no model retraining.
- 🚫 Avoids common issues like negative prompts being accidentally reinforcing the undesired concept.
- 🎯 Includes attention masking and token duplication to isolate effects to only where needed.

## Examples
<img width="1045" height="459" alt="image" src="https://github.com/user-attachments/assets/751e06db-bbd3-4c1b-b208-c384100efeea" />
The green prompt is the positive prompt, and the red text is the negative prompt. 


## Usage
You can clone this repo into your working folder, and execute the following code

```python
import torch
from src.pipeline import VSFStableDiffusion3Pipeline
pipe = VSFStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo",
    torch_dtype=torch.bfloat16,
).to("cuda")
prompt = "A poker table is set in the casino room, green felt stretched tight over the oval surface."
negative_prompt = "cards"
image_ours = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0, # This has to be 0
    num_inference_steps=8,
    scale=3.5,
    offset=0.1
).images[0].save("demo.png)
```

A demo notebook and comparsion with [NAG](https://github.com/ChenDarYen/Normalized-Attention-Guidance/tree/main) can be found in [demo.ipynb](demo.ipynb).

## To-do List
This to-do list will be listed in issues. If it is not assigned yet, feel free to assign it to yourself and contribute 
- [x] Add support for SD3.5-large-turbo 
- [ ] Add support for Flux-Dev and Flux-Schnell
- [ ] Add Comfy-UI custom node
- [ ] Add Web-UI demo
