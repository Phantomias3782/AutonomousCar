import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect
import cv2
from imutils.video.pivideostream import PiVideoStream


def detect(image):
#    camera = PiVideoStream().start()
 #   time.sleep(1)
  #  return_value, image = camera.read()
#
 #   fig = plt.figure(figsize=(20, 10))
  #  ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
   # plt.imshow(image)
    #ax.set_title("Input Image")
    #ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    # process image
    #plt.imshow(
    output=lanedetect.lane_finding_pipeline(image)
    
    # plot also processed image
  # ax.set_title("Output Image [Lane Line Detected]")
   # plt.show()

    #del(camera)
    return output