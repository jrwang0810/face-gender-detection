from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
import time

# OpenCV and base64
import cv2
import base64

# Declare a flask app
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# No caching at all for API endpoints. In order to force display the newest predicted image from local device.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

#model = MobileNetV2(weights='imagenet')
"""
face_classifier = cv2.CascadeClassifier(
    "/Users/jiuruwang/Documents/GitHub/face-AI/face web app - Jiuru Wang/haarcascades/haarcascade_frontalface_default.xml"
)
gender_classifier = load_model(
    "/Users/jiuruwang/Documents/GitHub/face-AI/faceai/classifier/gender_models/simple_CNN.81-0.96.hdf5", compile=False)
"""
face_classifier = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)
gender_classifier = load_model(
    "models/simple_CNN.81-0.96.hdf5", compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 2
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2


# Model saved with Keras model.save()
#MODEL_PATH = 'models/your_model.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, face_classifier, gender_classifier):
    #img = img.resize((224, 224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    gender_labels = {0: 'female', 1: 'male'}
    color = (255, 255, 255)

    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        face = img[(y):(y + h), (x):(x + w)]
        # print(face)
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        img = cv2.putText(img, gender, (x+5, y+80), font,
                          fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite("static/result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



    """
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    """
    # return img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request

        img = base64_to_pil(request.json)
        # img = img.resize((224, 224))
        #img = img[:,:,:,3]
        img = np.array(img)
        print(img.shape)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR )
        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        model_predict(img, face_classifier, gender_classifier)
        """
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        
        return jsonify(result=result, probability=pred_proba)
        """
        filepath = "static/result.jpg"
        # with open(filepath, "rb") as img_file:
        #     my_string = base64.b64encode(img_file.read())
        #     my_stringc = my_string.decode('utf-8')

        return jsonify({'img_url': filepath})

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
