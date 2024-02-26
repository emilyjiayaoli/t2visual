import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

TRANSFORMERS_CACHE = '/project_data/ramanan/emilyli/.sd-venv/.cache'

device = "cuda"
num_images_per_prompt = 1

# from ..constants import TRANSFORMERS_CACHE

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16, cache_dir=TRANSFORMERS_CACHE, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16, cache_dir=TRANSFORMERS_CACHE, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

prompt = "Anthropomorphic cat dressed as a pilot"
negative_prompt = ""

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20
)
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.half(),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images

#Now decoder_output is a list with your PIL images
# save images
for i, image in enumerate(decoder_output):
    image.save(f"stable-cascade_{i}.jpeg")