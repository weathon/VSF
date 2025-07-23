import sys
sys.path.append("../")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video

import argparse
parser = argparse.ArgumentParser(description="Run NAG Wan Video Generation")
parser.add_argument("--prompt", type=str, default="A group of Caribbean dancers performing at a carnival, wearing colorful costumes with feathers and sequins. The camera follows their energetic dance moves and the lively music.")
parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, distorted, low resolution, nnatural motion, unnatural lighting")
parser.add_argument("--video_id", type=int, default=0, help="Video ID for saving the output video")

args = parser.parse_args()

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
prompt = args.prompt
neg_prompt = args.neg_prompt

height = 480
width = 832
frames = 33

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
export_to_video(output[5:], f"videos/{args.video_id:03d}_none.mp4", fps=15)