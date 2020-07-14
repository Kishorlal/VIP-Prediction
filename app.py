"""
Description: Flask app for Movie review sentimental analysis

@author: Kishorlal
"""
import os
import numpy as np
import cv2

# Importing Keras library
from tensorflow.keras.models import load_model

# Importing Flask library
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Create flask object
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),r"templates"))

# Saved path of our model
MODEL_PATH = os.path.join(os.path.dirname(__file__),r"Celebrity_Prediction.h5")

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img=cv2.imread(img_path)
    resize=cv2.resize(img,(32,32))
    img=cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    img=np.array(img)
    img = np.reshape(img, (1, 32, 32, 1))
    preds = model.predict(img)
    preds=np.argmax(preds, axis=1)
    mapping = {0:"ELON-MUSK", 1:"JIM-CARREY", 2:"MOHANLAL", 3:"MSDHONI", 4:"TOMCRUISE"}
    if preds[0] in mapping:
        preds=mapping[preds[0]] 
    else:
        preds="Model is Not Trained for the input image"
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(image_path)
        preds = model_predict(image_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)