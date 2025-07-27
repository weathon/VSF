import gradio as gr
import numpy as np
import imageio

import torch
from diffusers import AutoencoderKLWan
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers import WanVACEPipeline
from diffusers.utils import export_to_video
import uuid

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    adapter_name="lora"
) 
pipe = pipe.to("cuda")
height = 480
width = 832
import os
os.makedirs("videos", exist_ok=True)
def generate_video(positive_prompt, negative_prompt, guidance_scale, bias, step, frames, seed, progress=gr.Progress(track_tqdm=False)):
    lambda total: progress.tqdm(range(total))
        
    print(f"Generating video with params: {positive_prompt}, {negative_prompt}, {guidance_scale}, {bias}, {step}, {frames}")
    pipe.set_adapters("lora", 0.6)
    prompt = positive_prompt
    neg_prompt = negative_prompt

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

    neg_len = neg_prompt_embeds.shape[1]
    pos_len = pos_prompt_embeds.shape[1]
    print(neg_len, pos_len)


    img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
    print(img_len)
    mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
    # mask[:, :, -neg_len:] = -torch.inf # this should be negative
    mask[:, :, -neg_len:] = bias

    for block in pipe.transformer.blocks: 
        block.attn2.processor = WanAttnProcessor2_0(scale=guidance_scale, neg_prompt_length=neg_len, attn_mask=mask)

    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt=neg_prompt,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=step,
        guidance_scale=0.0, 
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]
    path = f"videos/{uuid.uuid4().hex}.mp4"
    export_to_video(output[5:], path, fps=15)
    output_path = path
    with open(output_path.replace(".mp4", ".txt"), "w") as f:
        f.write(f"Positive Prompt: {positive_prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Steps: {step}\n")
        f.write(f"Frames: {frames}\n")
        f.write(f"Seed: {seed}\n")
    print(f"Video saved to {output_path}")
    return output_path

    
with gr.Blocks(title="Value Sign Flip Wan 2.1 Demo") as demo:
    gr.Markdown("# Value Sign Flip Wan 2.1 Demo \n All videos generated are saved and might be used as demo videos in the future. \n\n This demo is based on Wan 2.1 T2V model and uses Value Sign Flip technique to generate videos with different guidance scales and biases. More on [GitHub](https://github.com/weathon/VSF/blob/main/wan.md)")

    with gr.Row():
        pos = gr.Textbox(label="Positive Prompt", value="A chef cat and a chef dog with chef suit baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon.")
        neg = gr.Textbox(label="Negative Prompt", value="white dog")

    with gr.Row():
        guidance = gr.Slider(0, 5, step=0.1, label="Guidance Scale", value=1.5)
        bias = gr.Slider(0, 0.5, step=0.01, label="Bias", value=0.1)
        step = gr.Slider(6, 15, step=1, label="Step", value=10)
        frames = gr.Slider(31, 81, step=1, label="Frames", value=81)
        seed = gr.Number(label="Seed", value=0, precision=0)

    out = gr.Video(label="Generated Video")

    btn = gr.Button("Generate")
    btn.click(fn=generate_video, inputs=[pos, neg, guidance, bias, step, frames, seed], outputs=out)

demo.launch()