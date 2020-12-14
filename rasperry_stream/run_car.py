from camera import VideoCamera
from car_controll import controll_car
import time
import logging
import threading
from lane_detection import lanedetect_steer
from object_detection import detect_webcam

camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()
response = "fals"
def test(name):
    print("testwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    print(name)
    logging.info(name)
    time.sleep(3)
    return "something"

#frame = camera.get_frame()
object_thread = threading.Thread(target=detect_webcam, args=(1,))
while True:
    steering=0
    frame = camera.get_frame()
    try:
        frame2, steering=lanedetect_steer.lane_finding_pipeline(frame)
    except:
        print("steeringerror")
    if not object_thread.is_alive():
        #print("start threadS")
        object_thread = threading.Thread(target=detect_webcam, args=(frame,))
        #object_thread = threading.Thread(target=test, args=("babababa",))
        response = object_thread.start()            
    #print(object_thread)
    car.steer(steering)
#    except Exception as e:
 #       print("Error in detection")
  #      print(e)
    
