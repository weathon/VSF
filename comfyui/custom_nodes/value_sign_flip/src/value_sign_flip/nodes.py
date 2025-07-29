from inspect import cleandoc
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from transformers import T5EncoderModel, AutoTokenizer
from .pipeline import WanPipeline
from .processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video
import folder_paths
from transformers import AutoConfig
from safetensors.torch import load_file
from diffusers.hooks.group_offloading import apply_group_offloading

class VSF:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (folder_paths.get_filename_list("text_encoders"),),
                "wan": (folder_paths.get_filename_list("diffusion_models"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "cfg_lora": (folder_paths.get_filename_list("loras"),),
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}),
                "positive_prompt": ("STRING", {"default": "A cessna flying over a snowy mountain landscape, with a clear blue sky and fluffy white clouds. The plane is flying at a low altitude, casting a shadow on the snow-covered ground below.", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "wings", "multiline": True}),                
                "steps": ("INT", {"default": 12}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "frames": ("INT", {"default": 81}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": -0.1, "min": -1.0, "max": 0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "cfg_lora_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                # "cpu_offload": ("BOOLEAN", {"default": True, "label": "CPU Offload"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "sampling"

    def main(self, clip, wan, cfg_lora, vae, model_id, positive_prompt, shift, negative_prompt, steps, height, width, frames, scale, offset, seed, cfg_lora_scale):
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_single_file(
            folder_paths.get_full_path("vae", vae),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        transformer = WanTransformer3DModel.from_single_file(
            folder_paths.get_full_path("diffusion_models", wan),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder_state = load_file(
            folder_paths.get_full_path("text_encoders", clip),
            device="cpu", 
        )

        text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=None,
            config=AutoConfig.from_pretrained(model_id, subfolder="text_encoder"),
            state_dict=text_encoder_state,
            torch_dtype=torch.bfloat16,
        )
        # text_encoder = T5EncoderModel.from_pretrained(
        #     pretrained_model_name_or_path=model_id,
        #     subfolder="text_encoder",
        # )
        # if cpu_offload:
        #     onload_device = torch.device("cuda")
        #     offload_device = torch.device("cpu")
        #     apply_group_offloading(text_encoder,
        #         onload_device=onload_device,
        #         offload_device=offload_device,
        #         offload_type="block_level",
        #         num_blocks_per_group=4
        #     )
        #     transformer.enable_group_offload(
        #         onload_device=onload_device,
        #         offload_device=offload_device,
        #         offload_type="leaf_level",
        #         use_stream=True
        #     )

        pipe = WanPipeline(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=UniPCMultistepScheduler.from_pretrained(
                model_id, subfolder="scheduler", flow_shift=shift, 
            ),
        )
        lora_path = folder_paths.get_full_path("loras", cfg_lora)
        pipe.load_lora_weights(
            lora_path,
            adapter_name="lora",
        ) 

        pipe.to(device="cuda", dtype=torch.bfloat16)

        neg_prompt_embeds, _ = pipe.encode_prompt(
            prompt=negative_prompt,
            padding=False,
            do_classifier_free_guidance=False,
        )

        pos_prompt_embeds, _ = pipe.encode_prompt( 
            prompt=positive_prompt,
            do_classifier_free_guidance=False, 
            max_sequence_length=512 - neg_prompt_embeds.shape[1],
        )
        pipe.set_adapters("lora", cfg_lora_scale)



        neg_len = neg_prompt_embeds.shape[1]
        pos_len = pos_prompt_embeds.shape[1]
        print(neg_len, pos_len)

        img_len = (height//8) * (width//8) * 3 * (frames // 4 + 1) // 12
        print(img_len)
        mask = torch.zeros((1, img_len, pos_len+neg_len)).cuda()
        mask[:, :, -neg_len:] = offset

        for block in pipe.transformer.blocks:
            block.attn2.processor = WanAttnProcessor2_0(scale=scale, neg_prompt_length=neg_len, attn_mask=mask)

        prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)

        output = pipe(
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps, 
            guidance_scale=0.0, 
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pt",
        )[0].float()[0].permute(0, 2, 3, 1).contiguous()
        print(output.shape)
        
        return (output, )

NODE_CLASS_MAPPINGS = {
    "Example": VSF
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Value Sign Flip"
}
