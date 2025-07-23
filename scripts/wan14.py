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

old_prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)

old_neg_prompt = "low quality video, blurry, distorted, low resolution, weird motion"
neg_prompt = "low quality, blurry, distorted, low resolution, nnatural motion, unnatural lighting"
height = 480
width = 832
frames = 31 

prompt_len_ratio = len(pipeline.tokenizer.tokenize(neg_prompt, padding=False, truncation=True, max_length=512))/len(pipeline.tokenizer.tokenize(prompt, padding=False, truncation=True, max_length=512))
print("prompt_len_ratio", prompt_len_ratio)
# print(len(pipeline.tokenizer.tokenize(old_prompt, padding=False, truncation=True, max_length=512))/len(pipeline.tokenizer.tokenize(old_neg_prompt, padding=False, truncation=True, max_length=512)))
# 5.18/2.04=2.54

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
pipeline.set_adapters("lora", 0.4)



neg_len = neg_prompt_embeds.shape[1]
pos_len = pos_prompt_embeds.shape[1]
print(neg_len, pos_len)

img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
print(img_len)
mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -0.3

for block in pipeline.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=0.1/prompt_len_ratio, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

output = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames,
    num_inference_steps=6,
    guidance_scale=0.0, 
    generator=torch.Generator(device="cuda").manual_seed(8384),
).frames[0]
# save video with video_id 3 digits
export_to_video(output[5:], f"14b_vsf.mp4", fps=15)


mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -torch.inf 

for block in pipeline.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=0.8, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

output = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames,
    num_inference_steps=6,
    guidance_scale=0.0, 
    generator=torch.Generator(device="cuda").manual_seed(8384),
).frames[0]
# save video with video_id 3 digits
export_to_video(output[5:], f"14b_original.mp4", fps=15)