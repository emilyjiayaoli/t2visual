import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from ..base_model import BaseModel

# Assuming constants.py is in the same directory and contains TRANSFORMERS_CACHE
from ..constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

class SDXL_2_1(BaseModel):
    """
    SDXL_2_1
    This class is used to generate images from descriptions using the Stable Diffusion 2.1 model.
    https://huggingface.co/stabilityai/stable-diffusion-2-1
    """
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self.model_pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch_dtype, cache_dir=TRANSFORMERS_CACHE)
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self.model_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.model_pipe.scheduler.config)
        if torch.cuda.is_available(): # Use GPU if available
            print(f"Moving model to GPU... device {device}")
            self.model_pipe.to(device)

    def generate(self, text_prompt, folder_path="./", filename="sdxl-2-1-image.png", 
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

