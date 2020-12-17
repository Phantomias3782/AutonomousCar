import os
os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect_steer
import cv2


for image_path in list(os.listdir('./Flat_adjusted')):
    image = mpimg.imread(f'./Flat_adjusted/{image_path}')
    print(image_path)

    # plot input image
    fig = plt.figure(figsize=(20, 10),num=f'{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    try:
        picture = lanedetect_steer.lane_finding_pipeline(image)
    except Exception:
        print("Inputdaten Fehler")
        continue


    #picture = lanedetect.lane_finding_pipeline(image)
    plt.imshow(picture)

    # plot also processed image
    ax.set_title("Output Image [Lane Line Detected]") 
    plt.show()

    if 0xFF == ord('q'):
        break