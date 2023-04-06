from flask import Flask, request , render_template
from keras.utils import img_to_array
from defect_detection.config import STATIC_FOLDER_PATH , IMAGE_SIZE, RESIZE_FACTOR
from defect_detection.utils.prediction import get_predictions
from defect_detection.utils import helpers
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def show_prediction():

    if request.method == 'POST':
        product_type = str(request.form['product'])
        image = request.files['img']
        img = Image.open(image)

        img = img.resize(IMAGE_SIZE)
        probs, pt1, pt2 = get_predictions(img, product_type)

        img_name = helpers.draw_rectangle(img.resize(size=(8*RESIZE_FACTOR,8*RESIZE_FACTOR)), pt1, pt2)

        # probs = f"Defective : {probs[0]} <br> Good : {probs[1]}"

        return render_template('result.html', probs=probs, img_name = img_name)





if __name__ == '__main__':
    app.run(debug=True)