from keras.utils import img_to_array, load_img
from keras.backend import expand_dims
from keras.applications.vgg16 import preprocess_input

from defect_detection.config import IMAGE_SIZE


def prepare_input_img(img_path, IMAGE_SIZE=IMAGE_SIZE):

    img = load_img(path = img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img




