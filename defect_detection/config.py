import sys,os
from keras.models import load_model
import gdown

print("Running config.py ....")

IMAGE_SIZE = (256,256)

MODELS_PATH = os.path.join(os.getcwd(), 'defect_detection' , 'models')

STATIC_FOLDER_PATH = os.path.join(os.getcwd(), 'static')

RESIZE_FACTOR = 8

THRESHOLD = 0.7

MODEL_DRIVE_IDS = {
    'bangle' : '1GZ-0YMxLOaLnHz1hB8GJR3SkCQmuyOx3'
    }


def download_models():
    os.makedirs(MODELS_PATH, exist_ok=True)
    print("Downloading the models...")
    for model_name, id in MODEL_DRIVE_IDS.items():
        gdown.download(
            url = f'https://drive.google.com/uc?id={id}',
            output=f'{MODELS_PATH}/{model_name}.h5',
            quiet=False
        )
        print(f"Download complete for {model_name}.h5...")
    print("All downloads complete. ")


def get_saved_models(MODELS_PATH=MODELS_PATH):
    """
    Returns dictionary containing model for each product
    """
    saved_models = {}       # format : {'product_type' : model}

    model_files = os.listdir(MODELS_PATH)

    for model_file in model_files:
        model = load_model(os.path.join(MODELS_PATH, model_file))
        product_name = (model_file.split('.h5')[0]).upper()
        saved_models[product_name] = model

    return saved_models


download_models()  

SAVED_MODELS = get_saved_models(MODELS_PATH=MODELS_PATH)


