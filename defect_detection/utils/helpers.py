from keras.utils import img_to_array, load_img
from keras.backend import expand_dims
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model,Model
from skimage.transform import resize
from PIL import Image
import os
import numpy as np
from defect_detection.config import IMAGE_SIZE, RESIZE_FACTOR


def prepare_input_img(img, IMAGE_SIZE=IMAGE_SIZE):

    img = Image.open(img)
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


def get_feature_maps(img, model:Model) :
    # get the vgg layer from the model
    vgg_layer = model.get_layer(name='vgg16')

    # define model to extract the feature maps
    feature_map_model = Model(
        inputs=vgg_layer.inputs,
        outputs = vgg_layer.layers[-2].output)
    
    feature_maps = feature_map_model.predict(img)[0]

    return feature_maps


def process_final_feature_map(final_feature_map: np.array) :

    shape1 = final_feature_map.shape[0] * RESIZE_FACTOR
    
    # resize the feature map/img
    resized_img = resize(final_feature_map , (shape1, shape1))

    # normalize the image
    normalized_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min() )

    return normalized_img



