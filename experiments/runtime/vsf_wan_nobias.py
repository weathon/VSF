import sys
sys.path.append("../../")
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from vsfwan.pipeline import WanPipeline
from vsfwan.processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video
import json
with open("../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)
import time
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
total_time = 0
count = 0
import wandb
wandb.init(project="compute", name="vsf_wan")
torch.cuda.reset_peak_memory_stats()


for i in dev_prompts[:25]:
    prompt = i["prompt"]
    neg_prompt = i["missing_element"]
    
    height = 480
    width = 832
    frames = 81 
    # not using no grad will cause very high memory usage but flatten at the end, but seems like only in nobias settings? No it doesn't matter, it already has no grad in declarator
    # but why last time I run it it used so much RAMs? that was after I ran everything else, when program starts it already used 24GB, so it was uncleaned? but look at nvidia it is cleaned that is after restart is everything else correct? did not run that sperately so yes likely
    # wait it is 36 again. still about the no grad just saving differences? I think still about no grad, but why only in None?
    with torch.no_grad():
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
    mask = None

    for block in pipe.transformer.blocks:
        block.attn2.processor = WanAttnProcessor2_0(scale=0.8, neg_prompt_length=neg_len, attn_mask=mask)

    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

    start_time = time.time()
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt=neg_prompt,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=8,
        guidance_scale=0.0, 
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    end_time = time.time()
    total_time += (end_time - start_time)
    count += 1
    wandb.log({"time_per_image": total_time / count})
wandb.log({"peak_memory": torch.cuda.max_memory_allocated() / (1024 ** 3)})
