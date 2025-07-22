# to use this script, you need to modify diffusers/loaders/peft.py _SET_ADAPTER_SCALE_FN_MAPPING to include NagWanTransformer3DModel as a copy of WanTransformer3DModel
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from nag import NagWanTransformer3DModel
from nag import NAGWanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = NagWanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = NAGWanPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    adapter_name="lora"
) 
# pipe.set_adapters("lora", 0.5)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

prompt = "Static camera, a metal ball rolls on a smooth tabletop."
neg_prompt = "low quality video, blurry, distorted, low resolution, weird motion"
output = pipe(
    prompt=prompt,
    nag_negative_prompt=neg_prompt,
    guidance_scale=0.0,
    nag_scale=8,
    height=480,
    width=832,
    num_inference_steps=4,
    num_frames=81,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "nag.mp4", fps=15)
