from flask import Flask, render_template, Response, request
from camera import VideoCamera
from car_controll import controll_car
from lane_detection import lanedetect_steer
from object_detection import object_detection 
import cv2
import time
import threading
from datetime import datetime

pi_camera = VideoCamera(flip=False)
car = controll_car.Car()
app = Flask(__name__)

def save_frames(frame):
    now = datetime.now()    
    cv2.imwrite("./pics/"+str(now)+".jpg",frame)
    time.sleep(3)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/objects')
def index2():
    return render_template('index2.html')

def gen(camera):
    # save_frames_thread = threading.Thread(target=save_frames, args=(1,))
    # object_thread = threading.Thread(target=detect_webcam_delay, args=(frame,))
    while True:
        # get frame from VideoCamera-instance
        frame = camera.get_frame()

        try:
            # if not save_frames_thread.is_alive():
            #     save_frames_thread = threading.Thread(target=save_frames, args=(frame,))
            #     save_frames_thread.start()
            # if not object_thread.is_alive():
            #    object_thread = threading.Thread(target=detect_webcam_delay, args=(frame,))
            #    object_thread.start()            
            
            # Get steering input instruction from lanedetect_steer
            frame, canny, steering=lanedetect_steer.lane_detection(frame,"outdoor")
            # frame, canny, steering=lanedetect_steer.lane_detection(frame,"indoor")
            
            # Give the steering instruction from lanedetect_steer to the Car-instance
            car.steer(steering)
            time.sleep(0.0125)

        except Exception as e:
            print("Error in detection")
            print(e)

        # Convert the processed frame to show it in the browser
        ret,frame=cv2.imencode(".jpg",frame) 
        frame = frame.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2(camera):
    while True:
        # get frame from VideoCamera-instance
        frame = camera.get_frame()

        try:            
            # Highlight persons on frame with object_detection
            frame = object_detection.detection_on_image(frame)

        except Exception as e:
            print("Error in detection")
            print(e)
        
        # Convert the processed frame to show it in the browser
        ret,frame=cv2.imencode(".jpg",frame) 
        frame = frame.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(pi_camera),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
