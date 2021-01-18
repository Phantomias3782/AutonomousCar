from camera import VideoCamera
from car_controll import controll_car
from lane_detection import lanedetect_steer
from object_detection import object_detection

camera = VideoCamera(flip=False)
car = controll_car.Car()

# object_thread = threading.Thread(target=object_detection.detect_raspberry_cam_delay, args=(1,))
while True:
    frame = camera.get_frame()
    try:
        # if not object_thread.is_alive():
        #     object_thread = threading.Thread(target=object_detection.detect_raspberry_cam_delay, args=(frame,))
        #     object_thread.start()

        # Get steering input instruction from lanedetect_steer
        frame2, canny, steering=lanedetect_steer.lane_detection(frame,"outdoor")
        # frame2, canny, steering=lanedetect_steer.lane_detection(frame,"indoor")

        # Give the steering instruction from lanedetect_steer to the Car-instance
        car.steer(steering)

    except Exception as e:
	    print(e)
        print("steeringerror")