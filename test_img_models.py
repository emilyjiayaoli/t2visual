from models.t2image import get_model_class, print_all_model_names

# Dalle3   # Don't need GPU

MJ_SERVER_URL = 'https://midjourney-proxy-production-402a.up.railway.app/' # default. don't change if you're baiqi


def test_dalle(openai_api_key=OAI_KEY):
    print("Initializing DALLE...", end="")
    model = get_model_class('DALLE')(openai_api_key, version=3) # DALLE-3
    print("Done.")

    print("--Testing DALLE-3...", end="")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="dalle3-image-test.jpeg", download=True, size="1024x1024")
    print("Done. Image saved at", save_path)

    print("Initializing DALLE2...", end="")
    model = get_model_class('DALLE')(openai_api_key, version=2) # DALLE-2
    print("Done.")

    print("--Testing DALLE-2...", end="")    
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="dalle2-image-test.jpeg", download=True, size="512x512", version=2)
    print("Done. Image saved at", save_path)

def test_midjourney(host_url=MJ_SERVER_URL):
    print("Initializing Midjourney v5...", end="")
    args = {
        'version': 5.0,
    }
    model = get_model_class('Midjourney')(host_url, **args)
    print("Done.")

    print("--Testing Midjourney...", end="")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="mj-image-test.jpeg", download=True)
    print("Done. Image saved at", save_path)

def test_deepfloyd(): # Running on CPU is not supported

    print("Initializing DeepFloyd_I_XL_v1...", end="")
    model = get_model_class('DeepFloyd_I_XL_v1')()
    print("Done.")

    print("--Testing DeepFloyd...", end="")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="df-image-test.jpeg")
    print("Done. Image saved at", save_path)


def test_sdxl_turbo(): # Running on CPU is not supported
    print("Initializing SDXL_Turbo...", end="")
    model = get_model_class('SDXL_Turbo')()
    print("Done.")

    print("--Testing SDXL_Turbo...", end="")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="sdxl-image-test.jpeg")
    print("Done. Image saved at", save_path)

def test_sdxl_base(): # Running on CPU is not supported
    print("Initializing SDXL...", end="")
    model = get_model_class('SDXL_Base')()
    print("Done.")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="sdxl-base-image-test.jpeg")
    print("Done. Image saved at", save_path)

def test_sdxl_2_1(): # Running on CPU is not supported
    print("Initializing SDXL...", end="")
    model = get_model_class('SDXL_2_1')()
    print("Done.")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./imgs", filename="sdxl-2-1-image-test.jpeg")
    print("Done. Image saved at", save_path)


def test_all():
    # test_dalle()
    # # test_deepfloyd()
    # # test_midjourney()
    test_sdxl_turbo()
    test_sdxl_base()
    test_sdxl_2_1()


if __name__ == "__main__":
    print_all_model_names()
    test_all()
