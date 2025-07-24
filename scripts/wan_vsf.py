import sys
sys.path.append("../")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video

import argparse
parser = argparse.ArgumentParser(description="Run NAG Wan Video Generation")
parser.add_argument("--prompt", type=str, default="A group of Caribbean dancers performing at a carnival, wearing colorful costumes with feathers and sequins. The camera follows their energetic dance moves and the lively music.")
parser.add_argument("--video_id", type=int, default=0, help="Video ID for saving the output video")
parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, distorted, low resolution, nnatural motion, unnatural lighting")
args = parser.parse_args()


model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
# pipe.load_lora_weights(
#     "Kijai/WanVideo_comfy",
#     weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
#     adapter_name="lora"
# ) 
pipe.load_lora_weights(
    "lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE",
    weight_name="Wan2.1_T2V_1.3B_SelfForcing_DMD-FP16-LoRA-Rank32.safetensors",
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
frames = 81

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

img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
print(img_len)
mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
mask[:, :, -neg_len:] = -0.2

for block in pipe.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=1.9, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

output = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt=neg_prompt,
    height=height,
    width=width,
    num_frames=frames,
    num_inference_steps=12,
    guidance_scale=0.0, 
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]
# save video with video_id 3 digits
export_to_video(output[5:], f"videos/{args.video_id:03d}_vsf.mp4", fps=15)
