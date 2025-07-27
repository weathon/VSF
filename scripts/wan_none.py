import sys
sys.path.append("../")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video

import argparse

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    adapter_name="lora"
) 
pipe = pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

# prompt = "A chef cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The cat is wearing a chef suit"
# neg_prompt = "chef hat"
# prompts = [
#     "A knief cutting a tomato on a cutting board, the camera captures the knife's sharp edge slicing through the tomato's skin, revealing its juicy interior.",
#     "A lava river flowing through a volcanic landscape, dark rocky terrain. The camera captures the the flow of lava. The sky is dark with ash clouds.",
#     "A plane flying over a snowy mountain range, with the sun setting in the background. The camera captures the plane's silhouette against the colorful sky and the snow-covered peaks below.",
#     "A machine learning scientist working in a lab, analyzing data on a computer screen. The camera captures the scientist's focused expression and the complex algorithms displayed on the screen.",
#     "A pet running through a field of flowers, with the sun shining brightly. The camera captures the pet's joyful expression and the vibrant colors of the flowers.",
# ]
# neg_prompts = [
#     "wooden board, low quality, blurry, low resolution, weird motion",
#     "red hot, bright, glow, low quality, blurry, low resolution, weird motion",
#     "wings, low quality, blurry, low resolution, weird motion",
#     "male with glasses, low quality, blurry, low resolution, weird motion",
#     "dog, low quality, blurry, low resolution, weird motion",
# ]
prompts = [
    "A cat chef is cooking a delicious meal in a cozy kitchen, with the camera capturing the cat's focused expression and the vibrant colors of the ingredients. The pan is sizzling on the stove, and the cat is carefully adding spices to the dish.",
    "A laptop is on the table, playing a video of a cat. The laptop is silver and sleek, with a high-resolution screen. There are also some books and a cup of coffee on the table.",
]

neg_prompts = [
    "window, low quality, blurry, low resolution, weird motion",
    "keyboard, low quality, blurry, low resolution, weird motion"
]
for video_id, (prompt, neg_prompt) in enumerate(zip(prompts, neg_prompts)):
    height = 480
    width = 832
    frames = 81

    pipe.set_adapters("lora", 0.5)

    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=12,
        guidance_scale=0.0, 
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    export_to_video(output[5:], f"videos/{video_id:03d}_none.mp4", fps=15)
