import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import time
import imutils

def load_coco_names():
    "load names of coco text file"

    # create epmty list
    class_list = []

    # open coco txt file
    with open("coco.names", "r") as f:

        class_list = [line.strip() for line in f.readlines()]
    
    # return list of classes
    return class_list


def load_yolo(tiny = True):
    "load yolo network. option to load tiny or normal one."

    print("started loading yolov3...")

    # get yolov3. Normal verison is tiny due to performance, for single pictures also normal yolo is performant
    if tiny:

        net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    
    else:

        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # extract single layers of network
    layer_names = net.getLayerNames()

    # get output layers
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # print status
    print("...loaded volov3 sucessfully")

    # return
    return output_layers, net

def information_cal(outs, height, width):
    "calculate objects in images"

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5: # possibility to reset confidence

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return w, h, x, y, boxes, confidences, class_ids

def check_reaction(label):
    "if label in speified list send signal"

    # set list
    check_list = ["person", "car"]

    if label in check_list:

        # reaction
        print("Attention!")

        return True

    else:

        return False

def calculate_distance(image, object_width):
    "get image and calculate distance to object"

    # set marker attributes
    marker_width = 16
    marker_distance = 50

    # check if calibration is done
    if "focalLength" in globals():

        # calcuate distance
        distance = (marker_width * focalLength) / object_width

        # return distance
        return distance

    else:

        calibrate("./test_images/marker.jpg", marker_width, marker_distance)

        # calcuate distance
        distance = (marker_width * focalLength) / object_width

        # return distance
        return distance

def calibrate(image, marker_width, marker_distance):
    "calibrate first image from camera"

    # fixed parameters in cm

    image = mping.imread(image)

    # convert the image to grayscale, blur it, detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

	# find contures and keep largest (marker)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)

	# compute marker
    marker = cv2.minAreaRect(c)

    # check edged image
    # cv2.imwrite("gray_test.jpg", edged)
    
    # make global (db for one variable is oversized)
    global focalLength

    # calculate focalLength
    focalLength = (marker[1][0] * marker_distance) / marker_width

    # get distance to marker and print to check 
    distance = (marker_width * focalLength) / marker[1][0]
    print("marker distance ckeck", distance)

def information_draw(boxes, confidences, class_ids, class_list, img):

    # set Non-maximum Suppression and normal threshold
    threshold = 0.5
    nms_threshold = 0.4

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)

    # generate color palette
    colors = np.random.uniform(0, 255, size=(len(class_list), 3))

    # set font
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):

        if i in indexes:

            x, y, w, h = boxes[i]
            label = str(class_list[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y -5), font, 1, color, 2)

            # send information if specific object in image
            reaction = check_reaction(label)

            if reaction:

                # calculate distance
                distance = calculate_distance(img, w)

                print("distance to ", label, "is ", distance)

                # interface to car movement! stop under certain distance to object

            else:

                print("No action required")
    
    # return edited image
    return img

def detect_image(image_path, tiny=True):

    # set figure
    fig = plt.figure(figsize=(20, 10))

    ax2 = fig.add_subplot(1,2,1, xticks = [], yticks = [])

    # Load and show original image
    img_original = mping.imread(image_path)
    ax2.imshow(img_original)
    ax2.set_title("Original")

    # get dimensions
    height, width, channels = img_original.shape

    # preprocess image
    blob = cv2.dnn.blobFromImage(img_original, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # detect objects
    output_layers, net = load_yolo(tiny=tiny)
    class_list = load_coco_names()

    net.setInput(blob)
    outs = net.forward(output_layers)

    # calculate boxes and classifications
    w, h, x, y, boxes, confidences, class_ids = information_cal(outs, height, width)

    # draw boxes and classification
    img = information_draw(boxes, confidences, class_ids, class_list, img_original)

    # show edited image
    ax = fig.add_subplot(1,2,2, xticks = [], yticks = [])
    ax.set_title("Detected")
    ax.imshow(img)
    
    plt.show()

def detect_webcam(tiny=True):
    
    # get camera feed
    video_capture = cv2.VideoCapture(0)

    output_layers, net = object_detection.load_yolo(tiny=tiny)
    class_list = object_detection.load_coco_names()

    while True:

        # get frame
        re, frame = video_capture.read()

        height, width, channels = frame.shape

        # preprocess frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        
        #Detecting objects
        net.setInput(blob)
        outs = net.forward(output_layers)

        # calculate boxes and classifications
        w, h, x, y, boxes, confidences, class_ids = information_cal(outs, height, width)

        # draw boxes and classification
        frame = information_draw(boxes, confidences, class_ids, class_list, frame)

        # show feed
        cv2.imshow("Image",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_capture.release()

###############################################
import os
os.chdir("/Users/andreasmac/Documents/Github/AutonomousCar/object-detection")

# directory = "../lane_detection/google_img"
directory = "./test_images/"

# calibrate(directory+"marker.jpg", 16, 50)

detect_image(directory+"neuhauser-strasse-detail.jpg", tiny = False)

# for image_path in list(os.listdir(directory)):

#     try:
#         detect_image(directory+image_path, tiny = False)
#     except:
#         print("Failed with image: ", image_path)

# detect_webcam()
