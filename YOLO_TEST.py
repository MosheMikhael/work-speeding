# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:46:31 2018

@author: moshe.f
"""

#import cv2
import numpy as np
import cv2
import time 
import sys
# function to get the output layer names 
# in the architecture

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
images_adr = 'output/new/{:05d}.png'
classes_adr = 'input/for yolo/classes.txt'

v2_weights_adr = 'input/for yolo/yolov2.weights'
v2_config_adr  = 'input/for yolo/yolov2.cfg'
v2_tiny_weights_adr = 'input/for yolo/yolov2-tiny.weights'
v2_tiny_config_adr  = 'input/for yolo/yolov2-tiny.cfg'

v3_weights_adr = 'input/for yolo/yolov3.weights'
v3_config_adr  = 'input/for yolo/yolov3.cfg'
v3_tiny_weights_adr = 'input/for yolo/yolov3-tiny.weights'
v3_tiny_config_adr  = 'input/for yolo/yolov3-tiny.cfg'

# read class names from text file
classes = None
with open(classes_adr, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
yolo_version = '3'
net = None
if yolo_version == '2':
    net = cv2.dnn.readNetFromDarknet(v2_config_adr, v2_weights_adr)
elif yolo_version == '2-tiny':
    net = cv2.dnn.readNetFromDarknet(v2_tiny_config_adr, v2_tiny_weights_adr)
elif yolo_version == '3-tiny':
    net = cv2.dnn.readNet(v3_tiny_config_adr, v3_tiny_weights_adr)
elif yolo_version == '3':
    net = cv2.dnn.readNet(config=v3_config_adr, model=v3_weights_adr)

cam = cv2.VideoCapture(0)
ret, image = cam.read()
#image = cv2.imread(images_adr.format(0))
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
itt = 0
# create input blob 
#ret = True
while ret:
    t1 = time.time()
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
    net.setInput(blob)
# run inference through the network
# and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

# initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
# apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
# go through the detections remaining
# after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]   
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
#cv2.imwrite('input/for yolo/out.png', image)


## display output image   
    t2 = time.time()
    T = t2 -t1
    strTime = 'time per frame: {:02.05f} sec'.format(T)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 30)
    image = cv2.putText(image, strTime, bottom_left_corner_of_text, font, 
                                          1, (255, 0, 255), 2)
    
#    image = cv2.putText(image, strTime, (0, 0), fontScale=1)
    cv2.imshow("object detection by YOLOv{}".format(yolo_version), image)
#    cv2.imwrite('output/result/{}.png'.format(itt), image)
## wait until any key is pressed
    key = cv2.waitKey(5)
    ret, image = cam.read()
    itt += 1    
#    image = cv2.imread(images_adr.format(itt))
    if key == ord('q'):
        ret = False
#    
# # save output image to disk
#cv2.imwrite("object-detection.jpg", image)
#
## release resources
cv2.destroyAllWindows()
cam.release()
