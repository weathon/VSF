from inspect import cleandoc
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from transformers import T5EncoderModel, AutoTokenizer
from .pipeline import WanPipeline
from .processor import WanAttnProcessor2_0
from diffusers.utils import export_to_video
import folder_paths

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
                # "clip": (folder_paths.get_filename_list("text_encoders"),),
                # "wan": (folder_paths.get_filename_list("diffusion_models"),),
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}),
                "positive_prompt": ("STRING", {"default": "A cessna flying over a snowy mountain landscape, with a clear blue sky and fluffy white clouds. The plane is flying at a low altitude, casting a shadow on the snow-covered ground below.", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "wings", "multiline": True}),                
                "steps": ("INT", {"default": 12}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "frames": ("INT", {"default": 81}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "sampling"

    def main(self, model_id, positive_prompt, negative_prompt, steps, height, width, frames):
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        
        pipe.load_lora_weights(
            "Kijai/WanVideo_comfy",
            weight_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            adapter_name="lora"
        ) 
        pipe.to("cuda")
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
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=0.0, 
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        return output

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Example": VSF
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Value Sign Flip"
}
