import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from processor import WanAttnProcessor2_0
from pipeline import WanPipeline
# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. no ears"
neg_prompt = "no ears"#"low quality, blurry, bad lighting, out of focus, poorly drawn, ugly"
neg_prompt_embeds, _ = pipe.encode_prompt(
    prompt=neg_prompt,
    padding=False,
    do_classifier_free_guidance=False,
)

neg_len = neg_prompt_embeds.shape[1]
pos_prompt_embeds, _ = pipe.encode_prompt( 
    prompt=prompt,
    do_classifier_free_guidance=False,
    max_sequence_length=512 #here it has to be 512 for best quality
)

pos_len = pos_prompt_embeds.shape[1]
print(neg_len, pos_len)
mask = torch.zeros((1, 14040, pos_len)).cuda()
mask[:, :, -neg_len:] = -0.5 # this should be negative

# mask = None
for block in pipe.transformer.blocks:
    block.attn2.processor = WanAttnProcessor2_0(scale=0.5, neg_prompt_length=neg_len, attn_mask=mask)

prompt_embeds = torch.cat([pos_prompt_embeds], dim=1)

output = pipe(
    prompt_embeds=prompt_embeds, 
    negative_prompt=neg_prompt,
    height=480,
    width=832,
    num_frames=33,
    num_inference_steps=30,
    guidance_scale=5.0, 
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]
export_to_video(output, "output.mp4", fps=15)