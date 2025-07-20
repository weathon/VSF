import sys
sys.path.append("..")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
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
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=1.0)

# prompt = "A chef cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The cat is wearing a chef suit"
# neg_prompt = "chef hat"
prompt = "a cat running in the field, the cat is wearing a red scarf and a blue hat, the field is full of flowers and the sky is clear"
neg_prompt = "camera motion"

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
mask[:, :, -neg_len:] = 0.1

for block in pipe.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=1.8, neg_prompt_length=neg_len, attn_mask=mask)

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
export_to_video(output[5:], "vsf.mp4", fps=15)
