import os
os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect
import cv2

camera = cv2.VideoCapture(0)

for i in range(1):
    # get image from cam
    return_value, image = camera.read()

    # # resize image
    # width = 640
    # height = 480
    # dim = (width, height)
    # image = cv2.resize(image, dim)

    # plot input image
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    # process image
    plt.imshow(lanedetect.lane_finding_pipeline(image))

    # plot also processed image
    ax.set_title("Output Image [Lane Line Detected]")
    plt.show()

del(camera)