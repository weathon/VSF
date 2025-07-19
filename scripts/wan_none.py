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
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

# prompt = "A chef cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The cat is wearing a chef suit"
# neg_prompt = "chef hat"
prompt = "A person wearing a wingsuit launches from a plane high above the clouds, seamlessly transitioning into flying a lightweight glider. The water below reflects the golden light, and seagulls fly nearby. Realistic lighting, high detail, cinematic aerial perspective."
neg_prompt = "camera motion"
height = 480
width = 832
frames = 81

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
export_to_video(output[5:], "none.mp4", fps=15)
