"""
The file defines the DeepFloyd_I_XL_v1 class, which is used to generate images from text prompts by calling the DeepFloyd_I_XL_v1 API.
"""

import os
from ..constants import TRANSFORMERS_CACHE
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil

from ..base_model import BaseModel
import torch
import os

"""Model DeepFloyd-I-XL-v1: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0 """
class DeepFloyd_I_XL_v1(BaseModel):
    def __init__(self):
        print("Loading DeepFloyd-I-XL-v1 model...")
        # self.stage 1
        self.stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16, cache_dir=TRANSFORMERS_CACHE)
        # self.stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        self.stage_1.enable_model_cpu_offload()
        print("Finished loading stage 1.")
        # self.stage 2
        self.stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, cache_dir=TRANSFORMERS_CACHE)
        # self.stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        self.stage_2.enable_model_cpu_offload()
        print("Finished loading stage 2.")

        '''
        # self.stage 3
        safety_modules = {"feature_extractor": self.stage_1.feature_extractor, "safety_checker": self.stage_1.safety_checker, "watermarker": self.stage_1.watermarker}
        self.stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16, cache_dir='/project_data/ramanan/emilyli/.sd-venv/.cache')
        # self.stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        # self.stage_3.enable_model_cpu_offload()
        print("Finished loading stage 3.")

        if torch.cuda.is_available():
            print("Moving models to GPU...")
            self.stage_1.to("cuda")
            self.stage_2.to("cuda")
            # self.stage_3.to("cuda")
        else:
            print("Enabling CPU offload...")
            self.stage_1.enable_model_cpu_offload()
            self.stage_2.enable_model_cpu_offload()
            self.stage_3.enable_model_cpu_offload()
        '''

    def generate(self, text_prompt, seed=0, folder_path=None, filename=None, noise_level=100):

        save_path = os.path.join(folder_path, filename)
        if os.path.exists(save_path):
            print(f"Image already exists at {save_path}")
            return save_path
        
        prompt = text_prompt

        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt) # text embeds

        generator = torch.manual_seed(seed)

        image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        # pt_to_pil(image)[0].save("./if_self.stage_I.png")

        image = self.stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        pt_to_pil(image)[0].save(save_path)

        # image = self.stage_3(prompt=prompt, image=image, generator=generator, noise_level=noise_level).images
        # image[0].save(save_path)

        return save_path


