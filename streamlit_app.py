import streamlit as st
from flask import Flask, request , render_template
from keras.utils import img_to_array
from defect_detection.config import STATIC_FOLDER_PATH , IMAGE_SIZE, RESIZE_FACTOR
from defect_detection.utils.prediction import get_predictions
from defect_detection.utils import helpers
from werkzeug.utils import secure_filename
from PIL import Image
import os
st.write(
    "Hello "
)

name = st.text_input("Name", key='name')

st.write(f"Hello, dear {name}")