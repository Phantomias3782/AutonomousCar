import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect_parametertuning
import lanedetect
import cv2

#image = mpimg.imread(f'./flatlane_img/{image_path}')

def detect_on_frame(picture):
    fig = plt.figure(figsize=(20, 10),num='./test.jpg')
    print(type(fig))
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    picture = lanedetect.lane_finding_pipeline(image)
    plt.imshow(picture)

    # plot also processed image
    ax.set_title("Output Image [Lane Line Detected]")
    plt.show()
    return None