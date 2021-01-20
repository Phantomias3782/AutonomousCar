# Based on: https://github.com/EbenKouao/pi-camera-stream-flask
# Copyright smartbuild.io Licensed under MIT
# See license text at https://github.com/EbenKouao/pi-camera-stream-flask/blob/master/LICENSE
# Modified by Jan Brebeck
# Date: 20.01.21

import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np

class VideoCamera(object):
    # initialise rasperry pi camera module
    def __init__(self, flip = False):
        self.vs = PiVideoStream().start()
        self.flip = flip
        # wait for the camera to start
        time.sleep(2.0)
    
    # resetting VideoCamera-instance, when it gets dropped
    def __del__(self):
        # stop recording
        self.vs.stop()

    # method to correct the camera input, if the camera is upside-down
    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    # getting frames from rasperry pi camera module
    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        return frame
