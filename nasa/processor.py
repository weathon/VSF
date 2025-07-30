from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention


class NAGJointAttnProcessor2_0:
    def __init__(self, nag_scale: float = 1.0, nag_tau: float = 2.5, nag_alpha:float = 0.125):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        apply_guidance = self.nag_scale > 0 and encoder_hidden_states is not None
        if apply_guidance:
            origin_batch_size = len(encoder_hidden_states) - batch_size
            assert len(encoder_hidden_states) / origin_batch_size in [2, 3, 4]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if apply_guidance:
            batch_size += origin_batch_size
            if batch_size == 2 * origin_batch_size:
                query = query.tile(2, 1, 1, 1)
                key = key.tile(2, 1, 1, 1)
                value = value.tile(2, 1, 1, 1)
            else:
                query = torch.cat([query, query[origin_batch_size:2 * origin_batch_size]], dim=0)
                key = torch.cat([key, key[origin_batch_size:2 * origin_batch_size]], dim=0)
                value = torch.cat([value, value[origin_batch_size:2 * origin_batch_size]], dim=0)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if apply_guidance:
            hidden_states_negative = hidden_states[-origin_batch_size:]
            if batch_size == 2 * origin_batch_size:
                hidden_states_positive = hidden_states[:origin_batch_size]
            else:
                hidden_states_positive = hidden_states[origin_batch_size:2 * origin_batch_size]

            hidden_states_guidance = hidden_states_positive - hidden_states_negative * self.nag_scale
            # norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_positive.shape)
            # norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_guidance.shape)

            # scale = norm_guidance / norm_positive
            # hidden_states_guidance = hidden_states_guidance * torch.minimum(scale, scale.new_ones(1) * self.nag_tau) / scale

            # hidden_states_guidance = hidden_states_guidance * self.nag_alpha + hidden_states_positive * (1 - self.nag_alpha)

            if batch_size == 2 * origin_batch_size:
                hidden_states = hidden_states_guidance
            elif batch_size == 3 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance), dim=0)
            elif batch_size == 4 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance, hidden_states[2 * origin_batch_size:3 * origin_batch_size]), dim=0)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

