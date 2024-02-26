import os
from ..constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
from diffusers import AutoPipelineForText2Image
from ..base_model import BaseModel
import torch
import os


""" SDXL-Turbo
    This class is used to generate images from descriptions using the SDXL-Turbo model.
    https://huggingface.co/stabilityai/sdxl-turbo
"""
class SDXL_Turbo(BaseModel):
    def __init__(self, device="cuda", variant="fp16", torch_dtype=torch.float32):
        """
        log_folder_path: path to the folder where the logs will be saved. only saving api errors for now.
        usr_provided_prompt: if provided, it will be used as the prompt for the generation, excluding sample specific caption.
        """

        self.model_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch_dtype, variant=variant, cache_dir=TRANSFORMERS_CACHE)
        if torch.cuda.is_available(): # Use GPU if available
            print(f"Moving model to GPU... device {device}")
            self.model_pipe.to(device)
    
    def generate(self, text_prompt, folder_path="./", filename="sdxl-turbo-image.jpeg", 
                    num_inference_steps=1, guidance_scale=0.0):
        
        # torch.backends.cudnn.enabled = False

        # call model
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
            

