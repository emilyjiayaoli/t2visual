import json
import os
from models.t2image import get_model_class, print_all_model_names

prompts_path = "./data/t2v_prompts.json"

def load_prompts_dict(path):
    with open(path, "r") as f:
        return json.load(f)
    
def get_model(name):
    if name == "SDXL_Base":
        return get_model_class('SDXL_Base')()
    elif name == "SDXL_2_1":
        return get_model_class('SDXL_2_1')()
    else:
        raise ValueError(f"Model {name} not found")


def generate(model_folder_path="./", model_name="SDXL_Base"):

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    folder_path = os.path.join(model_folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model = get_model(model_name)
    
    log = {}
    print("Loading model...")
    prompts = load_prompts_dict(prompts_path)
    print("Done.")
    for prompt in prompts:
        print("Prompt:", prompt["prompt"])

        prompt_data = {}
        id = prompt["id"]
        prompt = prompt["prompt"]

        filename = f"{id}.jpeg"

        prompt_data["id"] = id
        prompt_data["prompt"] = prompt

        save_path = model.generate(text_prompt=prompt, 
                                   folder_path=folder_path, 
                                   filename=filename)
        
        if save_path is not None:
            prompt_data["image_path"] = save_path
            log[id] = prompt_data

    #update log.json
    with open(os.path.join(model_folder_path, "log.json"), "w") as f:
        json.dump(log, f, indent=4)
        

generate(model_folder_path="./SDXL_Base", model_name="SDXL_Base")
    


