import os,sys
from defect_detection.utils.helpers import prepare_input_img
from defect_detection.utils.prediction import get_predictions
from PIL import Image
from keras.utils import img_to_array

if __name__ == '__main__':
    img = "/home/varad/Work/Projects/Defect_detection/static/input_imgs/1_2021-11-17 01_03_42.834350.jpg"
    img = Image.open(img)
    img = img_to_array(img)
    #img = prepare_input_img(img)
    #preds = get_predictions(img, 'hexnut')

    #print(preds)
