# import os
# os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import lanedetect



# for image_path in list(os.listdir('./google_img')):
#     print(image_path)
#     fig = plt.figure(figsize=(20, 10))
#     image = mpimg.imread(f'./google_img/{image_path}')
#     ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
#     plt.imshow(image)
#     ax.set_title("Input Image")
#     ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])
#     plt.imshow(lanedetect.lane_finding_pipeline(image))
#     # ax.set_title("Output Image [Lane Line Detected]")
#     plt.show()





import os
os.chdir('/Users/Syman/Documents/Studij/Semester05/Seminar/AutonomousCar/lane_detection')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lanedetect_parametertuning
import lanedetect
import cv2


for image_path in list(os.listdir('./google_img')):
    print(image_path)
    #image = mpimg.imread(f'./google_img/{image_path}')
    image = cv2.imread(f'./google_img/{image_path}')
    
    # # resize image
    # width = 640
    # height = 480
    # dim = (width, height)
    # image = cv2.resize(image, dim)

    # change image df values from uint8 to float32
    #image = image.astype(float)
    
    
    # plot input image
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])

    # process image
    picture = lanedetect_parametertuning.lane_finding_pipeline(image)
    #picture = lanedetect.lane_finding_pipeline(image)
    plt.imshow(picture)

    # plot also processed image
    ax.set_title("Output Image [Lane Line Detected]")
    plt.show()