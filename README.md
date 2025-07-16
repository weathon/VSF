# VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip

Paper and benchmark coming soon

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
## Examples
<img width="1045" height="459" alt="image" src="https://github.com/user-attachments/assets/751e06db-bbd3-4c1b-b208-c384100efeea" />
The green prompt is the positive prompt, and the red text is the negative prompt. 

## To-do List
This to-do list will be listed in issues. If it is not assigned yet, feel free to assign it to yourself and contribute 
- [x] Add support for SD3.5-large-turbo 
- [ ] Add support for Flux-Dev and Flux-Schnell
- [ ] Add Comfy-UI custom node
- [ ] Add Web-UI demo
