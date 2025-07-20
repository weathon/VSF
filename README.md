# VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip
This project is experimental; please leave your feedback in issues or contact us. 
Email: email@weasoft.com


Paper and benchmark coming soon
## Web Demo for Wan 2.1 VSF
Wan 2.1 web demo [https://huggingface.co/spaces/weathon/VSF](https://huggingface.co/spaces/weathon/VSF)



## Introduction
This project introduces a new method called Value Sign Flip (VSF) that improves how image generation models handle negative prompts.

Problem: Modern few-step text-to-image models often struggle to properly exclude concepts described in negative prompts. Existing methods (CFG) either don’t work well or require heavy changes to the model (NegationCLIP).

Solution (VSF): We propose a lightweight technique that flips the value vector of negative prompt embeddings during attention. This cancels out unwanted features without retraining or needing access to classifier-free ⚡️.

Key Advantages:

- ⚡ Works with few-step and even single-step generation models (currently only supports SD3.5, Flux, and Wan), able to generate video with negative guidance in 30s. (480p, Wan 1.3B, 81 frames)
- 🔧 Requires no model retraining.
- 🚫 Avoids common issues like negative prompts being accidentally reinforcing the undesired concept.
- 🎯 Includes attention masking and token duplication to isolate effects to only where needed.

## News
- 📼 July 17, 2025: We now had experimental support for Wan 2.1
- 🖼️ July 16, 2025: We now support Flux Dev and Flux Schnell
- 🎨 July 15, 2025: We open sourced our repo and has support for SD3.5-large-turbo

## Examples
### SD3.5
<img width="1045" height="459" alt="image" src="https://github.com/user-attachments/assets/751e06db-bbd3-4c1b-b208-c384100efeea" />
This is an SD3.5 example; the green prompt is the positive prompt, and the red text is the negative prompt. 

### Flux
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/49068cac-e737-4f91-be95-b48d337e2e73" />
Positive Prompt: `a chef cat making a cake in the kitchen, the kitchen is modern and well-lit, the text on cake is saying 'I LOVE AI, the whole image is in oil paint style'`

Negative Prompt: `chef hat`

Scale: `3.5`


<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/be934907-14f7-44e0-bb5f-c59177cb80c0" />
Positive Prompt: `a chef cat making a cake in the kitchen, the kitchen is modern and well-lit, the text on cake is saying 'I LOVE AI, the whole image is in oil paint style'`

Negative Prompt: `icing`

Scale: `4`




[This video](media/flux_demo.mp4) shows a positive prompt of `a canadian winter landscape in the style of a 19th century painting` and negative prompt of `snow` at different scale, from 1 to 8.9 ([Code](flux_demo.py)). We can see as the scale increase the snow is decreasing. 

<!-- <video src="media/flux_demo.mp4" controls preload></video> -->

### Wan 2.1
**Our WAN examples are very exciting; however, due to file size, we put the examples in a separate file.** [wan.md](wan.md)

## Usage
You can clone this repo into your working folder, and execute the following code. We subjectively find that SD3.5 version is better at following negative prompt while Flux Schnell version has better quality. It seems like our method did not work well on Flux Dev. 

**Note: the CFG scale has to be set to 0 to use our method. **

### Wan WEb Demo
Clone the repo, and run `python3 app.py` will start a gradio interface for Wan.


### SD3.5-large-turbo
```python
import torch
from src.sd3_pipeline import VSFStableDiffusion3Pipeline
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
    generator=torch.Generator("cpu").manual_seed(19)
).images[0].save("demo.png")
```

A demo notebook and comparsion with [NAG](https://github.com/ChenDarYen/Normalized-Attention-Guidance/tree/main) can be found in [demo.ipynb](demo.ipynb).

### Flux Schnell
```python
import torch
from src.flux_pipeline import VSFFluxPipeline
import numpy as np
import imageio

pipe = VSFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

prompt = "a canadian winter landscape in the style of a 19th century painting"
image = pipe(
    prompt,
    negative_prompt="snow on the ground",
    guidance_scale=0.0,
    num_inference_steps=8,
    max_sequence_length=256,
    scale=6,
    generator=torch.Generator("cpu").manual_seed(19)
).images[0].save("demo.png")
```

### Flux Dev
(Our method doesn't seem to work on Flux Dev)
```python
import torch
from src.flux_pipeline import VSFFluxPipeline
import numpy as np
import imageio

pipe = VSFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "a bike on a snowy road in the style of a 19th century painting"
image = pipe(
    prompt,
    negative_prompt="wheels",
    guidance_scale=0.0,
    num_inference_steps=32,
    max_sequence_length=256,
    scale=8,
    generator=torch.Generator("cpu").manual_seed(19)
).images[0].save("demo.png")
```
### Wan2.1

Wan 2.1 does not have a complete pipeline yet, so the code is a bit long
```python
import torch
from diffusers import AutoencoderKLWan
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    adapter_name="lora"
) 
pipe = pipe.to("cuda")

# prompt = "A chef cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The cat is wearing a chef suit"
# neg_prompt = "chef hat"
prompt = "A cessna flying over a snowy mountain landscape, with a clear blue sky and fluffy white clouds. The plane is flying at a low altitude, casting a shadow on the snow-covered ground below. The mountains are rugged and steep, with patches of evergreen trees visible in the foreground."
neg_prompt = "trees"

neg_prompt_embeds, _ = pipe.encode_prompt(
    prompt=neg_prompt,
    padding=False,
    do_classifier_free_guidance=False,
)

pos_prompt_embeds, _ = pipe.encode_prompt( 
    prompt=prompt,
    do_classifier_free_guidance=False, 
    max_sequence_length=512 - neg_prompt_embeds.shape[1],
)
pipe.set_adapters("lora", 0.5)



neg_len = neg_prompt_embeds.shape[1]
pos_len = pos_prompt_embeds.shape[1]
print(neg_len, pos_len)
height = 480
width = 832
frames = 81

img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
print(img_len)
mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -0.2 # this should be negative

for block in pipe.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=1.7, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

output = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames + 1,
    num_inference_steps=12,
    guidance_scale=0.0, 
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]
export_to_video(output, "vsf.mp4", fps=15)
```


## To-do List
This to-do list will be listed in issues. If it is not assigned yet, feel free to assign it to yourself and contribute 
- [x] Add support for SD3.5-large-turbo 
- [x] Add support for Flux-Schnell
- [x] Add Wan2.1 support
- [x] Gradio Interface
- [ ] Add full Wan 2.1 work (non experimental LoRA)
- [ ] Make Flux-Dev work
- [ ] Add Comfy-UI custom node
- [ ] Add Web-UI demo

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weathon/VSF&type=Timeline)](https://www.star-history.com/#weathon/VSF&Timeline)
