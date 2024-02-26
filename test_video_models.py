from models.t2video import get_model_class, print_all_model_names


def test_zeroscope(): # Running on CPU is not supported
    print("Initializing ZeroScope...", end="")
    model = get_model_class('ZeroScope')()
    print("Done.")
    save_path = model.generate(prompt="Darth Vader is surfing on waves", folder_path="./", filename="zeroscope-video.mp4")
    print("Done. Video saved at", save_path)

def test_modelscope(): # Running on CPU is not supported
    print("Initializing ModelScope...", end="")
    model = get_model_class('ModelScope')()
    print("Done.")
    save_path = model.generate(text_prompt="A red apple on a table", folder_path="./", filename="modelscope-video.mp4")
    print("Done. Video saved at", save_path)


def test_all():
    test_zeroscope()
    test_modelscope()


if __name__ == "__main__":
    print_all_model_names()
    test_all()