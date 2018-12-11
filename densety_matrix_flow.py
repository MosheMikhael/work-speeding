# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:42:40 2018

@author: moshe.f
"""

import cv2
import numpy as np
import time
from collections import deque
import point_at_infinity as pai
import service_functions as sf
import draw
import video_make as vm
import sys

def id_to_ij(id, k):
    return id %k, id // k

def ij_to_id(i, j, k):
    return j * k + i

def avr(arr, v=-1):
    out = 0
    counter = 0
    for i in range(len(arr)):
        if not arr[i] == v:
            counter += 1
            out += arr[i]
    return out / counter

def make(video='C://Users//moshe.f//Desktop//TEST//calibration//23.mp4',
         m=100, k=100, nf=50, min_number_of_points=5, 
         out_data=None, out_pic=None):
    
    cam = cv2.VideoCapture(video)
    lengh = np.int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print(lengh)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
    
    _, frame1 = cam.read()
    y, x, z = frame1.shape
    
    # рабочая область
    x1, y1, x2, y2 = x // 5, y // 5, 4 * x // 5, 0.92 * y // 1
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    mask = [p1, p2]
    
    # ___________________
    times = deque(maxlen=lengh)
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
    
    sumX = np.zeros(m*k, dtype=np.float32)
    sumY = np.zeros(m*k, dtype=np.float32)
    sumX2 = np.zeros(m*k, dtype=np.float32)
    sumY2 = np.zeros(m*k, dtype=np.float32)
    Num = np.zeros(m*k, dtype=np.int)
    
    avr_x = np.zeros(m*k, dtype=np.float32)
    avr_y = np.zeros(m*k, dtype=np.float32)
    std_x2 = np.zeros(m*k, dtype=np.float32)
    std_y2 = np.zeros(m*k, dtype=np.float32)
    eps = np.zeros(m*k, dtype=np.float32)
    
    avr_eps = 0
    counter = 0
    # data collection
    for itt in range(nf):
        t0 = time.time()
        _, frame2 = cam.read()
        area1 = sf.get_box(frame1, mask)
        area2 = sf.get_box(frame2, mask)
        points2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(area1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(area2, cv2.COLOR_BGR2GRAY), points1, None, **lk_params)     
        for i in range(m*k):
            if st[i] == 1:
                addX = points2[i][0][0] - points1[i][0][0]
                addY = points2[i][0][1] - points1[i][0][1]
                Num[i] += 1
                sumX[i] += addX
                sumY[i] += addY
                sumX2[i] += addX ** 2
                sumY2[i] += addY ** 2
        frame1 = frame2
        tk = time.time()
        times.append(tk - t0)
        if itt % (nf // 10) == 0:
            T = (sum(times) / len(times)) * (nf - itt)
            t_min = int(T // 60)
            t_sec = T % 60
            print('{} | {} min {:.02f} sec'.format(itt, t_min, t_sec))        
    times.clear()
    
    # data analysise
    for i in range(m*k):
        t0 = time.time()
        if Num[i] < min_number_of_points:
            eps[i] = -1
        else:
            avr_x[i] = sumX[i] / Num[i]
            avr_y[i] = sumY[i] / Num[i]
            std_x2[i] = sumX2[i] / Num[i] - avr_x[i] ** 2
            std_y2[i] = sumY2[i] / Num[i] - avr_y[i] ** 2
            
            eps[i] = np.sqrt(std_x2[i] + std_y2[i])
            if np.isnan(eps[i]):
                sys.exit('Arg sqrt in eps is bad in step {}!!! arg = {}'.format(i, std_x2[i] + std_y2[i]))
        tk = time.time()
        times.append(tk - t0)
        if i % 10 == 0:
            T = (sum(times) / len(times)) * (m*k - i)
            t_min = np.int(T // 60)
            t_sec = T % 60
            print('calculate {} | {} min {} sec'.format(i, t_min, t_sec))
    times.clear()
    
    with open('trace/eps.txt', 'w') as f:
        for i in range(m*k):
            f.write('{}\n'.format(eps[i]))
            
    # average eps
    avr_eps = avr(eps, -1)
    #        print(' >>> {} <<< '.format(i))
    print('avr_eps = {}'.format(avr_eps))
    color = draw.black
    frame2 = draw.draw_rectangle(frame2, mask, color=draw.cyan)
    #frame2 = sf.get_box(frame2, mask)
    if out_pic is not None:
        for i in range(m*k):
            #print(i, m*k)
            t0 = time.time()
            th = 2
            if eps[i] > avr_eps:
                color = draw.red
            elif eps[i] == -1:
                color = draw.purple
            else:
                color = draw.green
            if np.isnan(eps[i]):
                th = 3
                color = draw.black
            pt1 = points1[i][0] + mask[0]
            #dpt = np.array([0, 0])
            #if not np.isnan(avr_x[i]) and not np.isnan(avr_y[i]):
            dpt = np.array([avr_x[i], avr_y[i]])
            pt2 = pt1 + dpt
            frame2 = draw.draw_point(frame2, pt1, radius=1, color=draw.blue)
            frame2 = draw.draw_point(frame2, pt2, radius=1, color=draw.blue)
            frame2 = draw.draw_arrow(frame2, pt2, pt1, color, thickness=th)
            #frame2 = draw.draw_text(frame2, pt1, text='({}|{})'.format(i // k, i % k), font_scale=0.25, line_type=1)
            tk = time.time()
            times.append(tk - t0)
            if i % (m*k // 10) == 0:
                T = (sum(times) / len(times)) * (m*k - i)
                t_min = np.int(T // 60)
                t_sec = T % 60
                print(' drawing: {} | {} min {} sec'.format(i, t_min, t_sec))        
        cv2.imwrite(out_pic, frame2)
    if out_data is not None:
        with open(out_data, 'w') as f:
            for i in range(m*k):
                line = '{}|{}|{}|{}|{}\n'.format(i // k, i % k, avr_x[i], avr_y[i], eps[i])
                f.write(line)
            
    #    out = frame2.copy()
    #    for i in range(N):
    #         pt1 = pts1[i] + mask[0]
    #         pt2 = pts2[i] + mask[0]
    #         out = draw.draw_point(out, pt1, radius=3)
    #         out = draw.draw_point(out, pt2, radius=3)
    #         out = draw.draw_arrow(out, pt1, pt2)
    #    cv2.imwrite('out/{}.jpg'.format(itt), out)
    print('done')
    return avr_x, avr_y, eps
        
if __name__ == '__main__':
    make(m=100, k=100, out_data = 'data.txt', out_pic='out.jpg')
    