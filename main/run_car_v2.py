from camera import VideoCamera
from car_controll import controll_car
import time
import logging
import threading
from lane_detection import lanedetect_steer
from object_detection import object_detection

camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()

object_detection_thread = threading.Thread(target=object_detection.detect_webcam_delay, args=(1,))
input("Hit Enter to run the car")
car.run()
object_output=None
outputcounter=0
while True:
    frame = camera.get_frame()
    try:
        if not object_detection_thread.is_alive():
            print("objectdetection started")
            object_detection_thread = threading.Thread(target=object_detection.detect_webcam_delay, args=(1,))
            object_output= object_detection_thread.start()
        if object_output != None:
            outputcounter+=1
            object_output=None
        print(outputcounter)
        frame2, canny, steering=lanedetect_steer.lane_detection(frame,"indoor")
        car.steer(steering)

    except Exception as e:
    	print(e)
        print("steeringerror")
