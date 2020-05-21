import flask
from flask import Flask, render_template, url_for, request
import pickle
import random
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
labels = ['airplane', 'car', 'eyeglasses', 'giraffe',
          'ice cream', 'jail', 'key', 'octopus', 'sheep', 'umbrella']
label_dict = dict(enumerate(labels))
with open(f'Model_10cat_revised.pkl', 'rb') as f:
    model = pickle.load(f)

label_dict = dict(enumerate(labels))

# Initializing the Default Graph (prevent errors)'''
graph = tf.get_default_graph()

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21


# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')
facts = dict()
facts['airplane'] = ['The Wright brothers invented and flew the first airplane in 1903', 'Airplanes typically fly at an altitude of around 35,000 feet',
                     'In 1986, a plane called Voyager flew all the way around the world without landing or refueling.']
facts['eyeglasses'] = ['According to studies, 25 percent of the world population has to wear prescription glasses or corrective lenses',
                       'Ben Franklin invented bifocal glasses (to see nearby and distant objects)', 'The first eyeglasses were made in Northern Italy, most likely in Pisa, by about 1290']
facts['jail'] = ['The vast majority of people sent to jail are legally presumed innocent',
                 'The earliest records of prisons come from the 1st millennia BC, located on the areas of mighty ancient civilizations of Mesopotamia and Egypt', 'Over 2.2 million people are currently in U.S. jails or prisons']
facts['key'] = ['The Guinness Book of World Records verified that Lisa Large of Kansas City, Missouri has the largest collection of keys. It took Lisa two years to collect and catalog her keys. As of 2013 when she was awarded the world record, Lisa owned 3,604 keys.', ' The first known mechanical key duplication machine dates back to 1917.',
                'It may come as no surprise, but the famous escape artist Harry Houdini started his career as a locksmith. Houdini worked in a locksmith’s shop at age 11 and the young apprentice quickly learned how to pick any lock available at the time. Without lock and key technology, the world may have never experienced the man who some consider to be the greatest illusionist who ever lived.']
facts['giraffe'] = ['They don’t sleep much. Most of them get around 10 minutes to two hours of sleep per day.',
                    'Giraffes used to be known as ‘camel leopards’, due to their tall structure and leopard-like pattern.The scientific name today has not changed – Camelopardalis', 'Their tongues can be as long as 45cm, allowing them to get their lunch from very tall trees!']
facts['umbrella'] = [' Majority of modern umbrellas are made in China. One city in China (Shangyu) has over thousand umbrella factories', 'Modern security agencies are known to modify umbrellas for their secret purposes',
                     'While people today associate umbrellas with rain, the roots of the word have to do with shade from the sun—umbrella stems from the Latin umbra for “shade, shadow”']
facts['sheep'] = ['Sheep have rectangular pupils that give them vision of around 300 degrees',
                  'In the Falkland Islands (South Atlantic Ocean archipelago), sheep outnumber humans at a ratio of 350:1', 'Sheep have an excellent memory. They can remember individual sheep and humans for years, that is why they have best friends!']
facts['octopus'] = ['Octopuses have three hearts', 'Octopus arms have a mind of their own. Two-thirds of an octopus’ neurons reside in its arms, not its head. As a result, the arms can problem solve how to open a shellfish while their owners are busy doing something else', 'They eat their arms when bored']
facts['ice cream'] = ['Sunni Sky’s homemade ice cream has a “cold sweat” ice cream with peppers so hot that you have to sign a waiver before they will sell it to you',
                      'Researchers have shown that ice cream causes people to feel safer and more comfortable. Ice cream sales tend to increase during times of economic recession', 'The United States is one of the top 3 countries in the world with the highest ice cream consumption']
facts['car'] = ['There are now three states that have legalized self-driving cars on the road: Nevada, Florida, and California. ',
                'The world’s first speeding ticket was issued in 1902 (for a scandalous speed of 45mph', 'There are 1 billion cars currently in use on earth. About 165,000 cars are produced every day']

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
            index = np.argmax(my_prediction)

            # Associating the index and its value within the dictionnary
            final_pred = label_dict[index]

    return render_template('User_input.html', prediction=final_pred, fact=facts[final_pred][random.randint(0, 2)])


if __name__ == '__main__':
    app.run(debug=True)
