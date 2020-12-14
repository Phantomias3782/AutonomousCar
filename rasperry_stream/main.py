from flask import Flask, render_template, Response, request
from camera import VideoCamera
from car_controll import controll_car
from lane_detection import lanedetect_steer
from object_detection import detect_webcam
import cv2
import os
import logging
import threading
import time


pi_camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()

# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here


def gen(camera):
    #get camera frame
    while True:
        frame = camera.get_frame()
        lane_thread = threading.Thread(target=lanedetect_steer.lane_finding_pipeline(frame), args=(1),)
        object_thread = threading.Thread(target=detect_webcam(frame), args=(1,))
        try:
            frame, steering=lane_thread.start()
            if not object_thread.is_alive():
                frame2 = object_thread.start()
            lane_thread.join()
            #print(steering)
            #car.steer(steering)
        except Exception as e:
            print("Error in detection")
            print(e)
        ret,frame=cv2.imencode(".jpg",frame) 
        frame = frame.tobytes()
        print(frame2)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
