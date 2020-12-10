#Modified by smartbuilds.io
#Date: 27.09.20
#Desc: This scrtipt script..

import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np
from lanedetect_steer import lane_finding_pipeline
from object_detection import detect_webcam


class VideoCamera(object):
    def __init__(self, flip = False):
        self.vs = PiVideoStream().start()
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        try:
            #frame2=detect_webcam(frame)
            frame2=lane_finding_pipeline(frame)
        except:
            print("Error in detection")
            frame2=frame
        ret, jpeg = cv2.imencode('.jpg', frame2)

        return jpeg