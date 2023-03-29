from defect_detection.config import SAVED_MODELS
from defect_detection.utils.helpers import prepare_input_img

def get_predictions(img_path, product_name):
    img = prepare_input_img(img_path)

    # get the model for the given product
    model = SAVED_MODELS[f"{product_name}_model"]

    preds = model.predict(img)

    
    return preds


