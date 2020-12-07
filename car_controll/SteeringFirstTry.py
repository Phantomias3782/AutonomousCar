import RPi.GPIO as IO


s.ChangeDutyCycle(straight)
input("mitte")
s.ChangeDutyCycle(full_left)
input("links")
s.ChangeDutyCycle(full_right)
input("rechts")
s.ChangeDutyCycle(0)
s.stop()

#Modified by smartbuilds.io
#Date: 27.09.20
#Desc: This scrtipt script..

import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np
from call_lanedetect_cam import detect
from object_detection import detect_webcam
import math


class Car():
    def __init__(self):
        IO.setwarnings(False)
        IO.setmode(IO.BOARD)
        IO.setup(35,IO.OUT)

        self.steering =IO.PWM(35,250)

        self.FULL_LEFT=41
        self.STRAIGHT=33
        self.FULL_RIGHT=20

    
        self.steering.start(0)

    def __del__(self):
        self.steering.stop()

    def steer(self,steeringrate):
        if steeringrate == 0:
            self.steerig.ChangeDutyCycle(self.STRAIGHT)
        elif steeringrate > 0 and steeringrate <= 1:
            self.steerig.ChangeDutyCycle(self.FULL_LEFT * steeringrate)
        elif steeringrate < 0 and steeringrate >= -1:
            steeringrate = abs(steeringrate)
            self.steerig.ChangeDutyCycle(self.FULL_RIGHT * steeringrate)
        else:
            print("Error in steeringadvise")
            return False
        return True