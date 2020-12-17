from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
import keras
import numpy
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
# img = cv2.resize(cv2.imread("test.jpg"),(150,150))
# img = img.reshape((1,) + img.shape)
# img2 = cv2.resize(cv2.imread("test2.jpg"),(150,150))
# img2 = img2.reshape((1,) + img2.shape)
# pred = loaded_model.predict(img)
# print(pred)
# pred2 = loaded_model.predict(img2)
# print(pred2)

def detect(frame):
    frame = cv2.resize(frame,(150,150))
    frame = frame.reshape((1,) + frame.shape)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        pred = loaded_model.predict(frame)
    print(pred)