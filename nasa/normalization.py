from typing import Optional, Tuple

import torch
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX


class TruncAdaLayerNorm(AdaLayerNorm):
    def forward(
            self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.forward_old(
            x, 
            temb[:batch_size] if temb is not None else None,
        )


class TruncAdaLayerNormContinuous(AdaLayerNormContinuous):
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.forward_old(x, conditioning_embedding[:batch_size])


class TruncAdaLayerNormZero(AdaLayerNormZero):
    def forward(
            self,
            x: torch.Tensor,
            timestep: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            hidden_dtype: Optional[torch.dtype] = None,
            emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        return self.forward_old(
            x,
            timestep[:batch_size] if timestep is not None else None,
            class_labels[:batch_size] if class_labels is not None else None,
            hidden_dtype,
            emb[:batch_size] if emb is not None else None,
        )


class TruncSD35AdaLayerNormZeroX(SD35AdaLayerNormZeroX):
    def forward(
            self,
            hidden_states: torch.Tensor,
            emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = hidden_states.shape[0]
        return self.forward_old(
            hidden_states,
            emb[:batch_size] if emb is not None else None,
        )
        