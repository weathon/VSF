
# Value Sign Flip for ComfyUI (Wan 2.1)

## Purpose

This node implements **Value Sign Flip (VSF)** for **negative guidance without CFG** in ComfyUI. It is designed for object removal in video generation (e.g., removing bike wheels), **not for quality improvement**. Avoid using prompts like "low quality" as negative—they may degrade results.


Example (generated using 1.3B, thus lower quality):
<img width="1502" height="850" alt="image" src="https://github.com/user-attachments/assets/575fe78a-a7e2-47cd-9cc9-3897ba59b54d" />

---

## Installation

Install this node like any other ComfyUI custom node.

---

## Usage

This is an all-in-one node. Connect it like any standard generation node. Provide a **positive prompt**, **negative prompt**, and the relevant configuration. The output is a list of images representing a video. Connect it to a `SaveWebp` node for preview/export.

No example workflow is provided due to the simplicity of the setup.

---

### Parameters

* **model\_id**: HuggingFace model ID for the *diffusers* version of the Wan model (e.g., `1.3B`, `14B`). Only config, tokenizer, and scheduler are loaded from here.
* **cfg\_lora**: Choose a CFG LoRA like
  [`Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/tree/main)
  or some others from [here](https://huggingface.co/Kijai/WanVideo_comfy/tree/main).
* **scale**: VSF scale. Starts at `0` (unlike CFG which starts at `1`).
* **offset**: Lower (more negative) values decrease the impact of the negative prompt outside the target area.
* **shift**: Flow shift during sampling.
* **cfg\_lora\_scale**: Scale of the CFG LoRA. Recommended: `0.4`–`0.7`.

---

## Limitations

* No in-node progress bar during denoising. Use `Ctrl + ~` to view progress in the terminal.
* Minimal customization; functionality is self-contained.
* CPU offloading is not supported.
* Requires FP16 text encoder. Use `umt5_xxl_fp16.safetensors`. FP8 models are unsupported.

---

## Feedback

This is an experimental tool. Feedback or suggestions are welcome via GitHub Issues.
