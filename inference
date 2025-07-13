import torch
from processor import JointAttnProcessor2_0
def inference(pipe, prompt, neg_prompt, seed=0, scale=3, offset=0, steps=8):
    (
        pos_prompt_embeds,
        _,
        pos_pooled_prompt_embeds,
        _,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        # padding=False
    )
    (
        neg_prompt_embeds,
        _,
        neg_pooled_prompt_embeds,
        _,
    ) = pipe.encode_prompt(
        prompt=neg_prompt,
        prompt_2=neg_prompt,
        prompt_3=neg_prompt,
        padding=False
    )
    
    neg_len = neg_prompt_embeds.shape[1]
    pos_len = pos_prompt_embeds.shape[1]
    
    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
    attn_mask = torch.zeros((1, 4096 + prompt_embeds.shape[1], 4096 + prompt_embeds.shape[1] + neg_len))
    attn_mask[:,-neg_len-pos_len:,-neg_len:] = -torch.inf #prompts cannot see -neg 
    attn_mask[:,:-neg_len,-2*neg_len:-neg_len] = -torch.inf # image and positive prompt cannot see neg
    attn_mask[:,-neg_len:,4096:4096+pos_len] = -torch.inf # neg cannot see positive prompt
    attn_mask[:,:4096,-neg_len:] -= offset # 0.08 image seeing less -neg
    
    attn_mask = attn_mask.cuda()

    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0(scale=scale, attn_mask=attn_mask, neg_prompt_length=neg_len)

    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
    image_ours = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pos_pooled_prompt_embeds,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    ).images[0]
    return image_ours
