import json
import os
from models.t2video import get_model_class, print_all_model_names

prompts_path = "./data/Sora_prompts.json"

def load_prompts_dict(path):
    with open(path, "r") as f:
        return json.load(f)
    
def get_model(name):
    if name == "ZeroScope":
        return get_model_class('ZeroScope')()
    elif name == "ModelScope":
        return get_model_class('ModelScope')()
    else:
        raise ValueError(f"Model {name} not found")


def generate(model_folder_path="./", model_name="ZeroScope"):

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    folder_path = os.path.join(model_folder_path, "data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model = get_model(model_name)
    
    log = {}
    print("Loading model...")
    prompts = load_prompts_dict(prompts_path)
    print("Done.")

    if os.path.exists(os.path.join(model_folder_path, "log.json")):
        with open(os.path.join(model_folder_path, "log.json"), "r") as f:
            log = json.load(f)
            
    for prompt in prompts:
        print("Id:", prompt["id"], "Prompt:", prompt["prompt"])

        prompt_data = {}
        id = prompt["id"]
        prompt = prompt["prompt"]

        filename = f"{id}.mp4"

        prompt_data["id"] = id
        prompt_data["prompt"] = prompt

        save_path = model.generate(prompt=prompt, 
                                   folder_path=folder_path, 
                                   filename=filename)
        
        save_path = os.path.join(folder_path, filename)

        if save_path is not None:
            prompt_data["video_path"] = save_path
            log[id] = prompt_data

        #update log.json
        with open(os.path.join(model_folder_path, "log.json"), "w") as f:
            json.dump(log, f, indent=4)
        
        
generate(model_folder_path="./generated/ModelScope", model_name="ModelScope")
    


