import sys,os
from keras.models import load_model

print("Running config.py ....")

IMAGE_SIZE = (256,256)

MODELS_PATH = os.path.join(os.getcwd(), 'defect_detection' , 'models')

STATIC_FOLDER_PATH = os.path.join(os.getcwd(), 'static')

RESIZE_FACTOR = 16

THRESHOLD = 0.7



def get_saved_models(MODELS_PATH=MODELS_PATH):
    """
    Returns dictionary containing model for each product
    """
    saved_models = {}       # format : {'product_type' : model}

    model_files = os.listdir(MODELS_PATH)

    for model_file in model_files:
        model = load_model(os.path.join(MODELS_PATH, model_file))
        product_name = model_file.split('.h5')[0]
        saved_models[product_name] = model

    return saved_models

   
SAVED_MODELS = get_saved_models(MODELS_PATH=MODELS_PATH)
