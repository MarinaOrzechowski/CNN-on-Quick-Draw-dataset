# CNN on Quick, Draw! dataset

https://draw-learn.herokuapp.com/

This is an intuitive web application that combines entertainment and leaning. You begin by drawing something, and the application will recognize the object. It will also provide you with some interesting/fun fact about that object.
Draw&Learn was created in Flask and it uses a trained convolutional neural network model to recognize the drawn objects. 

## How to run this app
(The following instructions apply to Windows command line.)

1. Clone repository, open a terminal to the app folder

`git clone https://github.com/MarinaOrzechowski/CNN-on-Quick-Draw-dataset.git`

`cd CNN-on-Quick-Draw-dataset`

2. Create and activate a new virtual environment (recommended) by running the following:

`virtualenv venv`

`venv\scripts\activate`

3. Install the requirements:

`pip install -r requirements.txt`

4. Run the app:

`python app.py`

You can run the app on your browser at http://127.0.0.1:5000

## Demonstration
![drawing_game_draft](https://user-images.githubusercontent.com/43459295/81029549-6835af80-8e53-11ea-80b9-4c59a722fdae.gif)
