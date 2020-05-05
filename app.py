import flask
from flask import Flask, render_template, url_for, request
import pickle
import os
import base64
import numpy as np
import cv2
import keras
from keras import backend
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import datasets
from keras import models
from keras import layers
from keras.utils import to_categorical

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
global graph
graph = tf.get_default_graph()
# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

#num_of_categories = 6

# Our dictionary
labels = ['car', 'ice cream', 'octopus', 'sheep', 'umbrella']
label_dict = dict(enumerate(labels))
with open(f'Model3.pkl', 'rb') as f:
    model = pickle.load(f)

label_dict = dict(enumerate(labels))

# Initializing the Default Graph (prevent errors)'''

#model = tf.keras.models.load_model('Model_50cat_89acc.h5')
graph = tf.get_default_graph()

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21


# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

# First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('User_input.html')

# Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    with graph.as_default():
        set_session(sess)

        if request.method == 'POST':

            final_pred = None
            # Preprocess the image : set the image to 28x28 shape
            # Access the image
            draw = request.form['url']
            # Removing the useless part of the url.
            draw = draw[init_Base64:]
            # Decoding
            draw_decoded = base64.b64decode(draw)
            image = np.asarray(bytearray(draw_decoded), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            # Resizing and reshaping to keep the ratio.
            resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            vect = np.asarray(resized)
            vect = vect.reshape(28, 28)
            vect = vect.reshape(1, 28, 28, 1)

            # Launch prediction
            my_prediction = model.predict(vect)
            print(my_prediction)
            # Getting the index of the maximum prediction
            #indecies = my_prediction.argsort()[0][-2:]
            index = np.argmax(my_prediction)
            print(label_dict)
            # print(indecies)
            # Associating the index and its value within the dictionnary
            #final_pred = [label_dict[indecies[1]], label_dict[indecies[0]]]
            final_pred = label_dict[index]

    return render_template('User_input.html', prediction=final_pred)


if __name__ == '__main__':
    app.run(debug=True)
