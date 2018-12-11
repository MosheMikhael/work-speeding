# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:41:14 2018

@author: moshe.f
"""

import sys
import numpy as np
import cv2
import densety_matrix_flow as dmf
import service_functions as sf
import collections
import draw
import time
def read(file):
    x = collections.deque()
    y = collections.deque()
    e = collections.deque()
    with open(file, 'r') as f:
        itt = 0
        for line in f:
            data = line.split('|')
            x.append(np.float32(data[2]))
            y.append(np.float32(data[3]))
            e.append(np.float32(data[4]))
            itt += 1
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(e, dtype=np.float32)
times = collections.deque()
m = 100
k = 100
N = m * k
video = 'C://Users//moshe.f//Desktop//TEST//calibration//23.mp4'
cam = cv2.VideoCapture(video)
_, frame1 = cam.read()
y, x, z = frame1.shape
    
    # рабочая область
x1, y1, x2, y2 = x // 5, y // 5, 4 * x // 5, 0.92 * y // 1
p1 = np.array([x1, y1])
p2 = np.array([x2, y2])
mask = [p1, p2]
    
    # ___________________
area1 = sf.get_box(frame1, mask)
y_, x_, z_ = area1.shape
    
    # greed points
points1 = np.zeros((m * k, 1, 2), dtype = np.float32)
yy0 = 0.02 * y_ // 1
dy = (y_ - 2 * yy0) // m
xx0 = 0.02 * x_ // 1
dx = (x_ - 2 * xx0) // k
for i in range(m*k):
    points1[i][0][0] = xx0 + dx * (i % k)
    points1[i][0][1] = yy0 + dy * (i // k)
#______________________
source_data = 'data.txt'
out_img = 'out.jpg'
template_x, template_y, template_eps = read(source_data)
avr_eps = dmf.avr(template_eps)

#x, y, eps = dmf.make(video, m, k, 70, 5, out_data='70.txt', out_pic='new.jpg')
x, y, eps = read('70.txt')
norm_x = np.array(x)
norm_y = np.array(y)
good = np.zeros(N, dtype=bool)
for i in range(N):
    if i % (N // 10) == 0:
        print(i, N)
    if template_eps[i] != -1 and eps[i] != -1:
        good[i] = True
        norm = np.sqrt(template_x[i] ** 2 + template_y[i] ** 2)
        norm_x[i] /= norm
        norm_y[i] /= norm
        

for i in range(N):
    if not good[i]:
        continue
    t0 = time.time()
    th = 2
    if eps[i] > avr_eps:
        color = draw.red
    elif eps[i] == -1:
        color = draw.purple
    else:
        color = draw.green
        
    pt1 = points1[i][0] + mask[0]
    dpt = np.array([norm_x[i], norm_y[i]])
    pt2 = pt1 + dpt
    frame1 = draw.draw_point(frame1, pt1, radius=1, color=draw.blue)
    frame1 = draw.draw_point(frame1, pt2, radius=1, color=draw.blue)
    frame1 = draw.draw_arrow(frame1, pt2, pt1, color, thickness=th)
    #frame2 = draw.draw_text(frame2, pt1, text='({}|{})'.format(i // k, i % k), font_scale=0.25, line_type=1)
    tk = time.time()
    times.append(tk - t0)
    if i % (N // 10) == 0:
        T = (sum(times) / len(times)) * (N - i)
        t_min = np.int(T // 60)
        t_sec = T % 60
        print(' drawing: {} | {} min {} sec'.format(i, t_min, t_sec))        
cv2.imwrite('norm.jpg', frame1)
#for i in range(N):
    
#dmf.make(video, m,k, 700, 5, out_data=source_data, out_pic=out_img)    
ex, ey = 0, 0
count = 0
for i in range(N):
    if good[i]:
        count += 1
        ex += norm_x[i]
        ey += norm_y[i]
ex /= count
ey /= count
print(ex, ey)