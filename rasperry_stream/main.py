from flask import Flask, render_template, Response, request
from camera import VideoCamera
from car_controll import controll_car
from lane_detection import lanedetect_steer
from object_detection import detect_webcam
import cv2
import time
import threading
import os


pi_camera = VideoCamera(flip=False) # flip pi camera if upside down.
car = controll_car.Car()

# App Globals (do not edit)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen(camera):
    object_thread = threading.Thread(target=detect_webcam, args=(1,))
    while True:
        frame = camera.get_frame()
        try:
            #if not object_thread.is_alive():
             #   object_thread = threading.Thread(target=detect_webcam, args=(frame,))
              #  object_thread.start()            
            #frame = detect_webcam(frame)
            frame, steering=lanedetect_steer.lane_finding_pipeline(frame)
            #print(steering)
            car.steer(steering)
        except Exception as e:
            print("Error in detection")
            print(e)
        ret,frame=cv2.imencode(".jpg",frame) 
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/objects')
def index():
    return render_template('index2.html') #you can customze index.html here

def gen2(camera):
    while True:
        frame = camera.get_frame()
        try:            
            frame = detect_webcam(frame)
        except Exception as e:
            print("Error in detection")
            print(e)
        ret,frame=cv2.imencode(".jpg",frame) 
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed2')
def video_feed():
    return Response(gen2(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
