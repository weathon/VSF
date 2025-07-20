# Value Sign Flip Comfy UI Implementation for Wan 2.1

Negative Guidance Without CFG

## Install
Please install this node same as any other nodes. 

## Usage 
The usage of this all-in-one node is very straightforward, you just input the positive prompt and negative prompt in it with basic config, it will output a list of images, and you can save them as videos. You can start with default hyperparameters and change as needed. 

## Limitations
This is very experimental, and it has many limitations
- It is just a wrapper of HuggingFace Diffusers, so it load from HF instead of ComfyUI model library, so it will (auto) download the model again if you do not have the HF version.
- It does not have a progress bar at the top of the node during de-noising
- This is an all-in-one node, and it does not provide much customization 

**I am very new to ComfyUI, so please give me any advice or feedback in the issues if you have them.**