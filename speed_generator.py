# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:42:42 2018

@author: moshe.f
"""

import numpy as np
import parser_gpx as prs
import matplotlib.pyplot as plt
import draw
import math
import os
import sys
import cv2
class Time:
    def __init__(self, h, m, s):
        self.h = h
        self.m = m
        self.s = s
    def get_time(self, sec):
        _s = self.s + sec
        _m = self.m + _s // 60
        _s = int(_s % 60)
        _h = self.h + int(_m // 60)
        _m = int(_m % 60)
        return _h, _m, _s
    def get_sec(self):
        return self.s + 60 * self.m + 3600 * self.h
    
def num_frame_to_sec(fr, n):
    return n / fr

def get_delta(a1, b1, a2, b2):
    return prs.haversine(a1, b1, a2, b2)

def get_bearing(lat1, lon1, lat2, lon2):
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    brng = math.degrees(math.atan2(y, x))
    return brng + 180

def get_speed(dist, time_1, time_2):
    v = float(3600 * dist / (time_1 - time_2))
    return v

def del_last_files(adress):
    '''
        Service function for removing ald output of script.
        
    :return: None.
    '''
    dlist = os.listdir(adress)
    for f in dlist:
        if f.split('.')[-1] == 'png':
            os.remove(adress + f)

def get_data_from_parsed_file(filename):
    ind = []
    lon = []
    lat = []
    dist = []
    v = []
    with open(filename, 'r') as file:
        t_prev = -1
        for line in file:
            p = line.split('|')
            tt = p[0].split(':')
            t_current = [int(tt[0]), int(tt[1]), int(tt[2])]
            lat_current = float(p[1])
            lon_current = float(p[2])
            dist_current = float(p[3])
            v_current = 0
            if t_prev != -1:
                t1 = 3600 * t_current[0] + 60 * t_current[1] + t_current[2]
                t2 = 3600 * t_prev[0] + 60 * t_prev[1] + t_prev[2]
                v_current = get_speed(dist_current, t1, t2)
            
            dist.append(dist_current)
            ind.append(t_current)
            lat.append(lat_current)
            lon.append(lon_current)
            v.append(v_current)     
                    
            t_prev = t_current
    return ind, lon, lat, dist, v

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#del_last_files('output/speed/')
filename = 'input/parsed-1750be37-5499-4c6e-9421-9bb15b277a94.txt'
cam = cv2.VideoCapture('input/1750be37-5499-4c6e-9421-9bb15b277a94.mp4')
frame_rate = 25
times, lon, lat, dist, speeds = get_data_from_parsed_file(filename)
frame_delay = 1360 # 685 + 27 * 25
K = 15_000

#MAP GENERATING:
pt0 = -K * np.array([min(lon), min(lat)])
map_ = np.zeros((270, 240, 3)) + draw.cyan
for k in range(len(lat)):
    dp = np.array([ K * lon[k], K  * lat[k]])
    pt = pt0 + dp
    map_ = draw.draw_point(map_, (pt[1], pt[0]), 1, thickness=1, color=draw.blue)

img = cam.read()[1]
print(img.shape)
img[50: 320,1630:1870] = map_
cv2.imwrite('map.png', img)
#sys.exit('DONE')
lat_g, lon_g, brng_g, speed_g, x = [], [], [], [], []
TIME = Time(13, 55, 3)
add_time = TIME.get_sec()
time_of_2_raw = -1
for i in range(frame_delay):
    lat_g.append(0)
    lon_g.append(0)
    brng_g.append(0)
    speed_g.append(0)
    x.append(i)

for i in range(frame_delay, 15000):
    k = i
    t = k / 25 + add_time
    x.append(i)
    if i % 100 == 0:
        print(i)
    ii = -1
    for o in range(1, len(times)):
        tt = times[o]
        TIME_temp = Time(tt[0], tt[1], tt[2])
        tt = TIME_temp.get_sec()
        dt = tt - t # разница между временем записи с индексом о и текущим времинем.
        if dt <= 0:
            ii = o
            
    lat_g.append(lat[ii])
    lon_g.append(lon[ii])
    speed_g.append(speeds[ii])
    brng_g.append(get_bearing(lat[ii], lon[ii], lat[ii - 1], lon[ii - 1]))
#sys.exit()

#fig1, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#ax1.plot(lat_g[685:], 'ro')
#ax2.plot(lon_g[685:], 'ro')
#ax3.plot(brng_g[685:], 'ro')
#ax4.plot(speed_g[685:], 'ro')
#fig1.savefig('output/new/data/lat.png')   # save the figure to file
#fig2.savefig('output/new/data/lon.png')   # save the figure to file
#fig3.savefig('output/new/data/brng.png')   # save the figure to file
#fig4.savefig('output/new/data/speed.png')   # save the figure to file
#plt.close(fig1)    # close the figure
#plt.close(fig2)    # close the figure
#plt.close(fig3)    # close the figure
#plt.close(fig4)    # close the figure
        
#sys.exit()
dpt = np.array([1630,   50])

v_current = 'No data'
lat_current = 'No data'
brgn = 'No data'
lon_current = 'No data'
classes_adr = 'input/for yolo/classes.txt'
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
net = cv2.dnn.readNet(config=v3_config_adr, model=v3_weights_adr)
Width = img.shape[1]
scale = 0.00392
Height = img.shape[0]

for i in range(15000):
    v_current = -1
    out = cam.read()[1]
    if i % 10 == 0:
        print(i)
    if not i < 1360:
        break
    blob = cv2.dnn.blobFromImage(out, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out_data in outs:
        for detection in out_data:
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
    for i_ in indices:
        i_ = i_[0]
        box = boxes[i_]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]   
        draw_bounding_box(out, class_ids[i_], confidences[i_], round(x), round(y), round(x+w), round(y+h))
        
    tmp_map = map_.copy()
    if i < frame_delay:
        out = draw.draw_text(out, (1250, 990), text='Speed      = {}'.format(v_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 990), text='Speed      = {}'.format(v_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 910), text='Latitude     = {}'.format(lat_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 910), text='Latitude     = {}'.format(lat_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 950), text='Longitude   = {}'.format(lon_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 950), text='Longitude   = {}'.format(lon_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 870), text='Bearing     = {}'.format(brgn), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 870), text='Bearing     = {}'.format(brgn), color=draw.red, font_scale=1.3, line_type=3)
        tmp_map = draw.draw_text(tmp_map, (70, 90), text='No', color=draw.red, font_scale=2, line_type=10)
        tmp_map = draw.draw_text(tmp_map, (30, 170), text='data!', color=draw.red, font_scale=2, line_type=10)
        
        tmp_map = draw.draw_text(tmp_map, (70, 90), text='No', color=draw.gold, font_scale=2, line_type=3)
        tmp_map = draw.draw_text(tmp_map, (30, 170), text='data!', color=draw.gold, font_scale=2, line_type=3)
        out[50: 320,1630:1870] = tmp_map
    elif i >= frame_delay:
        v_current = speed_g[i]
        lat_current = lat_g[i]
        lon_current = lon_g[i]
        brgn_current = brng_g[i]
        out = draw.draw_text(out, (1250, 990), text='Speed      = {:03.07f}kmph'.format(v_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 990), text='Speed      = {:03.07f}kmph'.format(v_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 910), text='Latitude     = {:02.07f}deg'.format(lat_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 910), text='Latitude     = {:02.07f}deg'.format(lat_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 950), text='Longitude   = {:02.07f}deg'.format(lon_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 950), text='Longitude   = {:02.07f}deg'.format(lon_current), color=draw.red, font_scale=1.3, line_type=3)
        out = draw.draw_text(out, (1250, 870), text='Bearing     = {:02.07f}deg'.format(brgn_current), color=draw.white, font_scale=1.3, line_type=8)
        out = draw.draw_text(out, (1250, 870), text='Bearing     = {:02.07f}deg'.format(brgn_current), color=draw.red, font_scale=1.3, line_type=3)
        
        dp = np.array([ K * lon_current, K  * lat_current])
        pt = pt0 + dp
        tmp_map = draw.draw_point(tmp_map, (pt[1], pt[0]), 3, thickness=2, color=draw.red)
        out[50: 320,1630:1870] = tmp_map
    cv2.imwrite('output/new/{:05d}.png'.format(i), out)        
        
        