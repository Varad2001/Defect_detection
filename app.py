from flask import Flask, request , render_template
from keras.utils import img_to_array
from defect_detection.config import STATIC_FOLDER_PATH
from defect_detection.utils.prediction import get_predictions
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
    
        preds = get_predictions(img=img, product_name=product_type)[0]

        print(preds)

        return f"<p>{preds}</p>"





if __name__ == '__main__':
    app.run(debug=True)