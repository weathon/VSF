# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L148

# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from .processor import JointAttnProcessor2_0

import torch
 
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
    
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class VSFStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        padding: bool = True
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length" if padding else False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
        padding: bool = True
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length" if padding else "longest",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds



    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
        padding: bool = True,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in all the text-encoders.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
                padding=padding,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
                padding=padding,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                padding=padding,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 0.0, 
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        scale: float = 3.0,
        offset: float = 0.08,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*):
                A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                with a scale of `1`.
            skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
            skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt,
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            pos_prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            do_classifier_free_guidance=False,
        )
        (
            neg_prompt_embeds,
            _,
            neg_pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt,
            prompt_3=negative_prompt,
            do_classifier_free_guidance=False,
            padding=False
        )
        prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
        
        neg_len = neg_prompt_embeds.shape[1]
        pos_len = prompt_embeds.shape[1]
        
        img_len = (height // 8 // self.transformer.config.patch_size) * (width // 8 //self.transformer.config.patch_size)
        
        prompt_embeds = torch.cat([prompt_embeds, neg_prompt_embeds], dim=1)
        attn_mask = torch.zeros((1, img_len + prompt_embeds.shape[1], img_len + prompt_embeds.shape[1] + neg_len))
        attn_mask[:,-neg_len-pos_len:,-neg_len:] = -torch.inf #prompts cannot see -neg 
        attn_mask[:,:-neg_len,-2*neg_len:-neg_len] = -torch.inf # image and positive prompt cannot see neg
        attn_mask[:,-neg_len:,img_len:img_len+pos_len] = -torch.inf # neg cannot see positive prompt
        attn_mask[:,:img_len,-neg_len:] -= offset # 0.08 image seeing less -neg
        
        attn_mask = attn_mask.to(device=device, dtype=prompt_embeds.dtype)

        processors_backup = []
        self.maps = []
        self.images = []
        for block in self.transformer.transformer_blocks:
            processors_backup.append(block.attn.processor)
            block.attn.processor = JointAttnProcessor2_0(scale=scale, attn_mask=attn_mask, neg_prompt_length=neg_len, maps=self.maps)

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)
                    
                    
                # latents_ = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                # image = self.vae.decode(latents_, return_dict=False)[0]
                # image = self.image_processor.postprocess(image, output_type=output_type)
                # self.images.append(image)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        for i, blocks in enumerate(self.transformer.transformer_blocks):
            blocks.attn.processor = processors_backup[i]
            
        return StableDiffusion3PipelineOutput(images=image)


