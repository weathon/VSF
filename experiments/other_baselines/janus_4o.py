import os
import sys
sys.path.append("..")
sys.path.append("../..")
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import argparse
parser = argparse.ArgumentParser(description="Run NAG sweep")
parser.add_argument("--eval_later", action="store_true", help="Run evaluation later")
args = parser.parse_args()
        
# Load model and processor
model_path = "FreedomIntelligence/Janus-4o-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,torch_dtype=torch.bfloat16
)
vl_gpt = vl_gpt.cuda().eval()

# Define text-to-image generation function
def text_to_image_generate(input_prompt, output_path, vl_chat_processor, vl_gpt, temperature = 1.0, parallel_size = 2, cfg_weight = 5):

    torch.cuda.empty_cache()

    conversation = [
            {
             "role": "<|User|>",
            "content": input_prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    prompt = sft_format + vl_chat_processor.image_start_tag

    mmgpt = vl_gpt

    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
 
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec


        return PIL.Image.fromarray(visual_img[0])

import json
import wandb
with open("../../prompts/test_prompts.json.new", "r") as f:
    dev_prompts = json.load(f)

import judge
import tqdm
def run():
    wandb.init(project="nag-sweep")
    score = np.array([0, 0, 0], dtype=float)
    total = 0
    os.makedirs("./output", exist_ok=True)
    for seed in range(2):
        for idx, i in enumerate(tqdm.tqdm(dev_prompts)):
            prompt = i["prompt"] + ", but with no " + i["missing_element"]
            image_output_path = f"./output/{idx}_{seed}.png"
            image = text_to_image_generate(prompt, image_output_path, vl_chat_processor, vl_gpt, parallel_size = 1)
            if not args.eval_later:
                delta = judge.vqa(image, i["question_1"], i["question_2"])
                score += delta
                total += 1
                from PIL import ImageDraw, ImageFont
                wandb.log({"pos_score_overall":score[0]/total, "neg_score_overall":score[1]/total, "quality_score_overall": score[2]/total,"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}"), 
                          "pos_score": delta[0], "neg_score": delta[1], "quality_score": delta[2]}) 
            else:
                wandb.log({"img": wandb.Image(image, caption=f"+: {i['prompt']}\n -: {i['missing_element']}")})

run()
# run(4, 0.125, 2.5)