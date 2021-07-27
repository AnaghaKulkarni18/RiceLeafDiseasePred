from __future__ import division, print_function

# coding=utf-8
import numpy as np
import pandas as pd
import os
import re
import glob
import sys

# For computer vision, we may not need it, but importing it anyways
import cv2

# keras 
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout 
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Flask utilities
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer


# dict = {'Bacterial leaf blight': 0, 'Brown spot': 1, 'Leaf smut': 2}

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'riceleafpred_modelv16.h5'

# Loading the trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    
    # x = np.true_divide(x, 255)
    
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
       
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    
    if preds==0:
        preds="This rice leaf has disease - Bacterial leaf blight"
    elif preds==1:
        preds="This rice leaf has disease - Brown spot"
    else:
        preds="This rice leaf has disease - Leaf smut"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make predictions
        preds = model_predict(file_path, model)
        os.remove(file_path) #removes file from the server after prediction has been returned
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)