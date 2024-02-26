# #https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# from constants import TRANSFORMERS_CACHE

# # pip install invisible_watermark transformers accelerate safetensors

# from diffusers import DiffusionPipeline
# import torch

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=TRANSFORMERS_CACHE)
# # pipe.to("cuda")

# # if using torch < 2.0
# # pipe.enable_xformers_memory_efficient_attention()

# prompt = "An astronaut riding a green horse"
# images = pipe(prompt=prompt).images[0]

import os
from diffusers import DiffusionPipeline
import torch
from ..base_model import BaseModel

# Assuming constants.py is in the same directory and contains TRANSFORMERS_CACHE
from ..constants import TRANSFORMERS_CACHE

class SDXL_Base(BaseModel):
    """
    SDXL_Base
    This class is used to generate images from descriptions using the SDXL-Base model.
    https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    """
    def __init__(self, device="cuda", variant="fp16", torch_dtype=torch.float16):
        # self.model_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype, use_safetensors=True, variant=variant, cache_dir=TRANSFORMERS_CACHE)
        self.model_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=TRANSFORMERS_CACHE)
        if torch.cuda.is_available(): # Use GPU if available
            print(f"Moving model to GPU... device {device}")
            self.model_pipe.to(device)

    def generate(self, text_prompt, folder_path="./", filename="sdxl-base-image.jpeg", 
                    num_inference_steps=50, guidance_scale=7.5):
        save_path = os.path.join(folder_path, filename)
        if os.path.exists(save_path):
            print(f"Image already exists at {save_path}")
            return save_path

        print("Generating image with caption: {}".format(text_prompt))
        image = self.model_pipe(prompt=text_prompt, 
                                num_inference_steps=num_inference_steps, 
                                guidance_scale=guidance_scale).images[0]

        # save image
        image.save(save_path)
    
        return save_path
