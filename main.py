import os,sys
from defect_detection.utils.helpers import prepare_input_img
from defect_detection.utils.prediction import get_predictions
from PIL import Image
from keras.utils import img_to_array

if __name__ == '__main__':
