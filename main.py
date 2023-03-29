import os,sys
from defect_detection.utils.helpers import prepare_input_img
from defect_detection.utils.prediction import get_predictions

if __name__ == '__main__':
    img = "/home/varad/Work/Projects/Defect_detection/static/1_2021-11-17 01_01_42.956235.jpg"
    img = prepare_input_img(img)
    preds = get_predictions(img, 'hexnut')

    print(preds)
