
"""
This file defines the Dalle class, which is used to generate images from text prompts by calling the DALL-E API.

Interface:
    - generate(text_prompt): generates an image using Dalle from the given text_prompt, returns the image url.
    - reset_api_key(new_api_key): resets the API key to a new one.
    - set_prompt(prompt): sets the prompt to be used for the generation.
    - get_dalle_prompt(text_prompt): returns the prompt to be used for the generation.
"""

from typing import Optional
from openai import OpenAI
from ..base_model import BaseModel
import os
import requests 

class DALLE(BaseModel):
    def __init__(self, openai_api_key:str, version:int=3, usr_provided_prompt:Optional[str]=None):
        """
        @param[in] usr_provided_prompt: if provided, this will be used as the prompt for the generation, excluding sample specific caption.
        """

        if version == 3 or version == 2:
            self.version = version
        else:
            raise ValueError("Version must be 2 or 3.")

        self.client = OpenAI(api_key=openai_api_key) 

        # Prompt
        if usr_provided_prompt:
            self.set_prompt(usr_provided_prompt)
        else: # Default prompt suggested by OpenAI: https://platform.openai.com/docs/guides/images/prompting
            dafault_prompt = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS. "
            dafault_prompt += "Generate a realistic image with text_prompt: "
            self.set_prompt(dafault_prompt)

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_dalle_prompt(self, text_prompt):
        return self.prompt + text_prompt

    def reset_api_key(self, new_api_key):
        self.client = OpenAI(api_key=new_api_key)

    def _call_dalle_api_helper(self, prompt, **kwargs):

        if "size" in kwargs: size = kwargs["size"] 
        else: size = "1024x1024"
        
        if "quality" in kwargs: quality = kwargs["quality"]
        else: quality = "standard"

        if "n" in kwargs: n = kwargs["n"]
        else: n = 1
        
        response = self.client.images.generate(
            model=f"dall-e-{self.version}",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        return response
    
    def generate(self, text_prompt:str, folder_path:str="./", filename:str="dalle-image.jpeg", download:bool=True, **kwargs):
        """ 
        Calls the DALL-E API to generate an image from a given text_prompt. 

        @param[in] text_prompt: the text_prompt to be used for the generation.
        @param[in] folder_path: the folder path where the image will be downloaded.
        @param[in] filename: the name of the downloaded image + filetype e.g 'image.jpg'.
        @param[in] download: if provided, the generated image will be downloaded to the specified folder.
        @returns the URL of the generated image,
                 None if an error occured,
        """
        if download:
            assert folder_path is not None, "folder_path must be provided when download is True."
            assert filename is not None, "filename must be provided when download is True."

        prompt = self.get_dalle_prompt(text_prompt)
        dalle_response = self._call_dalle_api_helper(prompt, **kwargs)

        self.recent_dalle_response = dalle_response

        if 'errors' in vars(dalle_response).keys():
            print("Error occured. Response:", dalle_response)
            return None

        image_url = dalle_response.data[0].url

        if download:
            return self.download_image(image_url, folder_path, filename)
        else:   
            print(f"Generated image: {image_url}")
            return image_url
    

    
    def download_image(self, image_url:str, folder_path:str, filename:str):
        """
        Saves an image from a given URL to a specified file path.

        :param image_url: URL of the image to download.
        """
        assert folder_path is not None, "folder_path must be provided when download is True."
        assert filename is not None, "filename must be provided when download is True."
        assert image_url is not None, "image_url must be provided when download is True."

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_path = os.path.join(folder_path, filename)

        # if os.path.exists(save_path): # optional: Check if the image already exists
        #     print(f"Image already exists at {save_path}")
        #     return save_path
        
        response = requests.get(image_url) # Send a GET request to the image URL

        # Return image if the request was successful
        if response.status_code == 200:
            # Save the image
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return save_path
        else:
            print(f"Failed to download the image. Status code: {response.status_code}")
            return None
        