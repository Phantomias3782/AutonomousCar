import os
os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/main/lane_detection')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect_steer
import cv2

location = 'outdoor'

for image_path in list(os.listdir(f'./lane_detection_data/{location}')):
    image = mpimg.imread(f'./lane_detection_data/{location}/{image_path}')
    print(image_path)

    print('input picture in call: ' + str(type(image)))

    try:
        picture, canny, steering = lanedetect_steer.lane_detection(image, location)
        print('output picture in call: ' + str(type(picture)))
    except Exception:
        print("Berechnungsfehler")
        continue

    # plot input image
    fig = plt.figure(figsize=(20, 10),num=f'{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(canny)
    ax.set_title("canny transformation")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    #picture = lanedetect.lane_finding_pipeline(image)
    plt.imshow(picture)

    # plot also processed image
    ax.set_title(f"Output Image - Steering: {steering}") 
    plt.show()

    if 0xFF == ord('q'):
        break