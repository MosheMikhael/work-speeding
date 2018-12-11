# -*- coding: utf-8 -*-
"""
Experiment with density flaw

@author: moshe.f
"""
import time
import collections
import cv2
import numpy as np
data = 'C:/Users/moshe.f/Desktop/TEST/calibration/23.mp4'
#data = 'test1.mp4'
cap = cv2.VideoCapture(data)
ret, frame1 = cap.read()
N = 0
itt = N
for i in range(N):
    ret, frame1 = cap.read()
    print('{}/{}'.format(i, N))

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
y, x, z = frame1.shape
hsv[...,1] = 255
max_frame_index = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
times = collections.deque(maxlen=max_frame_index-itt)
print('Let\'s do it!')
while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    t0 = time.time()
    path = 'exp/{}.jpg'.format(itt)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
 #   cv2.imshow('frame2',bgr)
    out = np.zeros((y, 2*x, 3))
    out[:y, :x] = bgr
    out[:y, x:2*x] = frame2
    cv2.imwrite(path, out)
    tk = time.time()
    times.append(tk - t0)
    T = (sum(times) / len(times)) * (max_frame_index - itt)
    t_min = T // 60
    t_sec = T % 60
    print('{:.0f}min {:02.02f}sec\t{} frame of {}'.format(t_min, t_sec, itt, max_frame_index))
    prvs = next
    itt += 1    
cap.release()
print(sum(times) / len(times))