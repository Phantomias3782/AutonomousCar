import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import time
import imutils
import os

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
    print("...loaded yolov3 sucessfully")

    # return
    return output_layers, net

def information_cal(outs, height, width):
    "calculate objects in images"

    # init lists and set confidence treshhold
    confidence_treshold = 0.5
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # check if object is detected about given confidence
            if confidence > confidence_treshold:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # calcluate width and height of box
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # calculate x and y coordinates of box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # save informations about boxes, containing confidence and class
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def check_reaction(label):
    "if label in speified list send signal"

    # set list
    check_list = ["person", "car"]

    if label in check_list:

        # reaction
        print("Potential action required!")

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
        distance = round((marker_width * focalLength) / object_width, 2)

        # return distance
        return distance

    else:

        calibrate("./test-images/marker.jpg", marker_width, marker_distance)

        # calcuate distance
        distance = round((marker_width * focalLength) / object_width, 2)

        # return distance
        return distance

def calibrate(image, marker_width, marker_distance):
    "calibrate first image from camera"

    # load image
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
    distance = round((marker_width * focalLength) / marker[1][0], 2)
    print("marker distance ckeck", distance)

def information_draw(boxes, confidences, class_ids, class_list, img):

    # set Non-maximum Suppression and normal threshold
    threshold = 0.5
    nms_threshold = 0.4

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)

    # generate color palette
    colors = np.random.uniform(0, 255, size=(len(class_list), 3))

    # set font and other settings
    font = cv2.FONT_HERSHEY_PLAIN
    rec_width = 3
    txt_height = 3
    text_width = 3

    for i in range(len(boxes)):

        if i in indexes:

            x, y, w, h = boxes[i]
            label = str(class_list[class_ids[i]])
            full_label = label + ", " + str(round(confidences[i] * 100, 2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, rec_width)
            cv2.putText(img, full_label, (x, y -5), font, txt_height, color, text_width)

            # send information if specific object in image
            reaction = check_reaction(label)
            if reaction:

                # calculate distance
                distance = calculate_distance(img, w)

                print("distance to ", label, "is ", distance, "cm")

                # interface to car movement! stop under certain distance to object

            else:

                print("No action required")
    
    # return edited image
    
    return img

def detect_image(image_path, tiny=True):

    # set figure
    fig = plt.figure(figsize=(20, 10))

    ax2 = fig.add_subplot(1,2,1, xticks = [], yticks = [])

	# load and show original image
    image_path=os.path.abspath(os.getcwd())+image_path[1:]
    img_original = mping.imread(image_path)
    ax2.imshow(img_original)
    ax2.set_title("Original")

    # get dimensions
    height, width, channels = img_original.shape

    # preprocess image
    blob = cv2.dnn.blobFromImage(img_original, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # load network
    output_layers, net = load_yolo(tiny=tiny)

    # load coco list
    class_list = load_coco_names()

    # detect objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # calculate boxes and classifications
    boxes, confidences, class_ids = information_cal(outs, height, width)

    # draw boxes and classification
    img = information_draw(boxes, confidences, class_ids, class_list, img_original)

    # show edited image
    ax = fig.add_subplot(1,2,2, xticks = [], yticks = [])
    ax.set_title("Detected")
    ax.imshow(img)
    plt.show()


output_layers, net = load_yolo(tiny=True)
class_list = load_coco_names()

print("finished loading yolo")

def detect_webcam(frame,tiny=True):
    # load coco list
    
    height, width, channels = frame.shape

    # preprocess 
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    
    # detect objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # calculate boxes and 
    boxes, confidences, class_ids = information_cal(outs, height, width)

    # draw boxes and classification
    frame = information_draw(boxes, confidences, class_ids, class_list, frame)
    return frame
