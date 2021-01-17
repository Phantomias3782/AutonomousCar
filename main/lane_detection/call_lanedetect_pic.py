import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect_steer
import cv2
from timeit import default_timer as timer
import numpy as np

# set indoor or outdoor depending on test environment
location = 'outdoor'

time = []

for image_path in list(os.listdir(f'./lane_detection_data/{location}')):
    # start timer for speed calculation of lane detection
    start = timer()

    # call lane detection picture vise for all pictures in folder
    image = mpimg.imread(f'./lane_detection_data/{location}/{image_path}') 
    try:
        picture, canny, steering = lanedetect_steer.lane_detection(image, location)
    except Exception:
        continue

    # plot outputs
    fig = plt.figure(figsize=(20, 10),num=f'{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(canny)
    ax.set_title("canny transformation")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])
    plt.imshow(picture)
    ax.set_title(f"Output Image - Steering: {steering}") 
    plt.show()

    # end timer, save required time
    end = timer()
    time.append(float(end - start))
    if 0xFF == ord('q'):
        break

# calculate average time of lane detection data pipeline
print('durchschnittliche Zeit: ' + str(np.mean(time)))