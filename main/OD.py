"""A demo to classify Raspberry Pi camera stream."""
import argparse
import time

import numpy as np
import os
import datetime

#import edgetpu.detection.engine
import cv2
from PIL import Image

def detect(frame):
    model = './mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    label = './labels.txt'

    with open(label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    # initialize open cv
    # camera = cv2.VideoCapture(0)
    # ret = camera.set(3,IM_WIDTH)
    # ret = camera.set(4,IM_HEIGHT)
    
    IM_WIDTH = 640
    IM_HEIGHT = 480
    input = cv2.resize(frame, (IM_WIDTH,IM_HEIGHT))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,IM_HEIGHT-10)
    fontScale = 1
    fontColor = (255,255,255)  # white
    boxColor = (0,0,255)   # RED?
    boxLineWidth = 1
    lineType = 2
    
    annotate_text = ""
    annotate_text_time = time.time()
    time_to_show_prediction = 1.0 # ms
    min_confidence = 0.20
    
    # initial classification engine
    #engine = edgetpu.detection.engine.DetectionEngine(args.model)

    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model)
    # interpreter.allocate_tensors()

    # # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    # interpreter.invoke()

    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (IM_WIDTH,IM_HEIGHT))
    frame_expanded = np.expand_dims(frame, axis=0)
    # ret, img = camera.read()
    input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB color space
    img_pil = Image.fromarray(input)
    
    # results = engine.DetectWithImage(img_pil, threshold=min_confidence, keep_aspect_ratio=True,
    #                     relative_coord=False, top_k=5)
    results = DetectWithImage(img_pil, threshold=min_confidence, keep_aspect_ratio=True,
                         relative_coord=False, top_k=5)

    if results :
        for obj in results:
            print("%s, %.0f%% %s %.2fms" % (labels[obj.label_id], obj.score *100, obj.bounding_box, elapsed_tf_ms * 1000))
            box = obj.bounding_box
            coord_top_left = (int(box[0][0]), int(box[0][1]))
            coord_bottom_right = (int(box[1][0]), int(box[1][1]))
            cv2.rectangle(img, coord_top_left, coord_bottom_right, boxColor, boxLineWidth)
            annotate_text = "%s, %.0f%%" % (labels[obj.label_id], obj.score * 100)
            coord_top_left = (coord_top_left[0],coord_top_left[1]+15)
            cv2.putText(img, annotate_text, coord_top_left, font, fontScale, boxColor, lineType )
        print('------')
    else:
        print('No object detected')

    cv2.putText(frame, annotate_text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    # out.write(img)

    return frame
