from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_data_generator(image_paths, is_training=False):
    while True:
        batch_images = []
        
        for image_path in image_paths:
            image_path="./test-images2/"+image_path
            image = my_imread(image_path)
            image = img_preprocess(image)
            batch_images.append(image)
            
        return np.asarray(batch_images)

#X_valid = ["./test-images/1.jpg","./test-images/2.jpg","./test-images/3.jpg","./test-images/4.jpg"]
# X_valid=os.listdir("./test-images2")

# X = image_data_generator(X_valid)


with tf.Graph().as_default():
    model = load_model('lane_navigation.h5')
    def detect(frame):
        frame = np.asarray([img_preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))])
        Y_pred = model.predict(frame)
        cv2.putText(frame,Y_pred)
        return frame
