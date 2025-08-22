import torch
import sys
sys.path.append("../")
import numpy as np
from diffusers import WanTransformer3DModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.bfloat16
)

pipeline.to("cuda")

pipeline.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    adapter_name="lora"
)

prompt = "There is a crab at the bottom of the ocean that looks like a rock, it is surrended by real rocks and camouflaged into the background because the rock and the crab has the same color and texture. The crab is far and small and ran across the screen."
neg_prompt = "the animal is clearly visible, standing out, easy to spot, obvious, distinct, different color"



height = 480
width = 832
frames = 81

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

pipeline.set_adapters("lora", 0.9)
neg_prompt_embeds = neg_prompt_embeds#.mean(1, keepdim=True)


neg_len = neg_prompt_embeds.shape[1]
pos_len = pos_prompt_embeds.shape[1]
print(neg_len, pos_len)

img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
print(img_len)
mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -0.2

for block in pipeline.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=0.8, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1).to("cuda")

output = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames,
    num_inference_steps=12,
    guidance_scale=0.0,
    generator=torch.Generator(device="cuda").manual_seed(80),
).frames[0]
export_to_video(output[5:], f"14b_lora.mp4", fps=15)