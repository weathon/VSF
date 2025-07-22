# pip install ftfy
import torch
import sys
sys.path.append("../")
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)
transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    adapter_name="lora"
) 

pipeline.to("cuda")

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""

neg_prompt = "low quality video, blurry, distorted, low resolution, weird motion"
height = 480
width = 832
frames = 33 


neg_prompt_embeds, _ = pipeline.encode_prompt(
    prompt=neg_prompt,
    padding=False,
    do_classifier_free_guidance=False,
)

pos_prompt_embeds, _ = pipeline.encode_prompt( 
    prompt=prompt,
    do_classifier_free_guidance=False, 
    max_sequence_length=512 - neg_prompt_embeds.shape[1],
)
pipeline.set_adapters("lora", 0.45)



neg_len = neg_prompt_embeds.shape[1]
pos_len = pos_prompt_embeds.shape[1]
print(neg_len, pos_len)

img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
print(img_len)
mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -0.15

for block in pipeline.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=1, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

output = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames,
    num_inference_steps=8,
    guidance_scale=0.0, 
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]
# save video with video_id 3 digits
export_to_video(output[5:], f"14b_vsf.mp4", fps=15)
