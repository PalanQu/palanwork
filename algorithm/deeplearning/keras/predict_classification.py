from keras.models import load_model
import numpy as np
import random
from keras.preprocessing.image import img_to_array, load_img
import os

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model.
BASE_PATH = "/Users/qujiabao/code/palanwork"
visualization_model = load_model(os.path.join(BASE_PATH, "model"))

# Let's prepare a random input image of a cat or dog from the training set.
img_path = os.path.join(BASE_PATH, "998.jpg")

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
print(successive_feature_maps)
