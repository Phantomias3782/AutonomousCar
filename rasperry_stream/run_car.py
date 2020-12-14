from camera import VideoCamera
from car_controll import controll_car
import time
import logging
import threading
from lane_detection import lanedetect_steer
camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()
response = "fals"


frame = camera.get_frame()
object_thread = threading.Thread(target=detect_webcam(frame), args=(1,))
while True:
    frame = camera.get_frame()
    try:
        frame2, steering=lanedetect_steer.lane_finding_pipeline(frame)
        #print(steering)
        if not object_thread.is_alive():
            object_thread = threading.Thread(target=detect_webcam(frame), args=(1,))
            response = object_thread.start()            
        print(response)
        car.steer(steering)
    except Exception as e:
        print("Error in detection")
        print(e)
    
