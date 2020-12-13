from camera import VideoCamera
from car_controll import controll_car
import time
from lane_detection import lanedetect_steer
camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()

while True:
    frame = camera.get_frame()
    try:
        frame2, steering=lanedetect_steer.lane_finding_pipeline(frame)
        print(steering)


        car.steer(steering)
    except Exception as e:
        print("Error in detection")
        print(e)
    
