from camera import VideoCamera
from car_controll import controll_car
import time
import logging
import threading
from lane_detection import lanedetect_steer

camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()
counter=0
steering_mid=0
while True:
    steering=0
    frame = camera.get_frame()
    try:
        frame2, steering=lanedetect_steer.lane_finding_pipeline(frame)
        if steering != None:
            steering_mid+=steering
        else:
            counter-=1
        print(steering)

    except Exception as e:

       	print(e)
        print("steeringerror")
    if counter%5==0:
        car.steer(steering_mid/5)
        steering_mid=0
    counter+=1
    
