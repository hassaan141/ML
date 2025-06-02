# train a hand written nn classifier

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 


# this gets the existing dataset from keras instead of downloading a csv
mnist = tf.keras.datasets.mnist

# training data, validation to test the data on data it has not seen before
# load data function already setup

# is the handwritten data and y is the label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# now we have to normalize it so its between 0 - 1. 0 - 255 grayscale becomes 0 - 1
x_train = tf.keras.utils.normalize(x_trian, axis=1)
x_test = tf.keras.utils.normalize(x_trian, axis=1)

# imports the basic sequential neural networks
model = tf.keras.models.Sequential()

# flatten the input layer, turns the grid into a 784 line
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# now is the dense layer, where each neuron is connected to every previous neuron. Using relu instead of a sigmond function, 
# relu: negative values become 0 and only keeps positive values, Sigmond is better when outputs need to be probability
# 
model.add(tf.keras.layers.Dense(128, activation='relu'))

#do this again, pattern layer
model.add(tf.keras.layers.Dense(128, activation='relu'))

#ouput layer, this will ouput a number between 0 - 9
#softmax adds up everything, so it will be a confidence probablity, each number will have a value between 0 - 9. Gives us the probabilty
model.add(tf.keras.layers.Dense(128, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#now we have to train the model, pass in the fit function and train the data

model.fit(x_train, y_train, epoch=3)

model.save('handwritten.model')