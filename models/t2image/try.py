from diffusers import DiffusionPipeline
import torch


TRANSFORMERS_CACHE = '/Users/emily/Desktop/code repos/winogroundv2/synthetic-bench-master/t2visual-share/t2v-venv/.cache'
# HF_HOME = '/project_data/ramanan/emilyli/.sd-venv/.cache'

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=TRANSFORMERS_CACHE)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]

# save image
images.save("sdxl-base-image.jpeg")