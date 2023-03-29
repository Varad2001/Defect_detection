from flask import Flask, request , render_template
from keras.utils import img_to_array
from defect_detection.config import STATIC_FOLDER_PATH
from defect_detection.utils.prediction import get_predictions
from werkzeug.utils import secure_filename
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
        img_save_path = os.path.join(STATIC_FOLDER_PATH,
                                      'input_imgs',
                                      secure_filename(image.filename))
        
        image.save(img_save_path)
        preds = get_predictions(img_path=img_save_path, product_name=product_type)[0]

        print(preds)

        return f"<p>Predictions:{preds}</p>"





if __name__ == '__main__':
    app.run(debug=True)