# reduce number of images for each category to 20000 from >100000
# I only did it so I could upload the folder to Google Drive and work with it at Google Colab.
# Before reduction the folder was 37 Gb, now it is 5 Gb
# The result is in folder C:/Users/mskac/machineLearning/movie_guess/data/testData/
import os
import numpy as np

files = os.listdir(
    "C:/Users/mskac/machineLearning/CNN-on-Quick-Draw-dataset/data/full")

reduced = 15000
for file in files:
    x = np.load(
        "C:/Users/mskac/machineLearning/CNN-on-Quick-Draw-dataset/data/full/" + file)
    x = x[:reduced]
    np.save(
        "C:/Users/mskac/machineLearning/CNN-on-Quick-Draw-dataset/data/short/" + file, x)
