from camera import VideoCamera
from car_controll import controll_car
import time
import logging
import threading
from lane_detection import lanedetect_steer
from object_detection import detect_webcam,detect_webcam_delay

camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()
while True:
    frame = camera.get_frame()
    try:
        if not object_thread.is_alive():
            print("objectdetection started")
            object_thread = threading.Thread(target=detect_webcam_delay, args=(frame,))
            object_thread.start()
        frame2, steering=lanedetect_steer.lane_finding_pipeline(frame)
        car.steer(steering)
        #print(steering)

    except Exception as e:
       	print(e)
        #print("steeringerror")