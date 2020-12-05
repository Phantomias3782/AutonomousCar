import os
os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect



for image_path in list(os.listdir('./google_img')):
    print(image_path)
    fig = plt.figure(figsize=(20, 10))
    image = mpimg.imread(f'./google_img/{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])
    plt.imshow(lanedetect.lane_finding_pipeline(image))
    # ax.set_title("Output Image [Lane Line Detected]")
    plt.show()