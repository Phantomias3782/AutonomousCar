import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_coco_names():

    # create epmty list

    class_list = []

    # open coco txt file
    with open("coco.names", "r") as f:

        class_list = [line.strip() for line in f.readlines()]
    
    # return
    return class_list


def load_yolo(tiny = True):

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

def information_draw(boxes, confidences, class_ids, class_list, img):

    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(class_list), 3))
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_list[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y -5), font,
            1, color, 2)
    
    # return edited image
    return img

def detect_image(image_path):

    # set figure
    fig = plt.figure(figsize=(20, 10))

    ax2 = fig.add_subplot(1,2,1, xticks = [], yticks = [])

    # Load and show original image
    img_original = cv2.imread(image_path)
    ax2.imshow(img_original)
    ax2.set_title("Original")

    # get dimensions
    height, width, channels = img_original.shape

    # preprocess image
    blob = cv2.dnn.blobFromImage(img_original, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # detect objects
    output_layers, net = load_yolo(tiny=False)
    class_list = load_coco_names()

    net.setInput(blob)
    outs = net.forward(output_layers)

    # calculate boxes and classifications
    w, h, x, y, boxes, confidences, class_ids = information_cal(outs, height, width)

    # draw boxes and classification
    img = information_draw(boxes, confidences, class_ids, class_list, img_original)

    #cv2.imshow("Image",img)

    # show edited image
    ax = fig.add_subplot(1,2,2, xticks = [], yticks = [])
    ax.set_title("Detected")
    ax.imshow(img)
    
    plt.show()

    print("end")

import os
os.chdir("/Users/andreasmac/Documents/Github/AutonomousCar/object-detection")

# detect_image("./test-images/neuhauser-strasse-detail.jpg")
# detect_image("../lane_detection/google_img/google1.jpg")