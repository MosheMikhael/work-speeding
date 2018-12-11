# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:20:17 2018

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
def action(method):
    video = 'test1.mp4'
    
    cam = cv2.VideoCapture(video)
    lengh = np.int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print(lengh)
    _, frame1 = cam.read()
    y, x, z = frame1.shape
    x1, y1, x2, y2 = x // 3, 3*y//5, 2*x//3, 0.94 * y // 1
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    mask = [p1, p2]
    
    times = deque(maxlen=lengh)
    #times = deque(maxlen=lengh)
    
    for itt in range(lengh):
        t0 = time.time()
        _, frame2 = cam.read()
        if not _:
            break
        frame2 = draw.draw_rectangle(frame2, mask, color=draw.cyan)
        area1 = sf.get_box(frame1, mask)
        area2 = sf.get_box(frame2, mask)
        y_, x_, z_ = area1.shape
        print(y_, x_)
        points1 = np.zeros((100, 2), dtype = np.float)
        yy0 = 0.02 * y_ // 1
        dy = (y_ - 2 * yy0) // 10
        xx0 = 0.02 * x_ // 1
        dx = (x_ - 2*xx0) // 10
        for i in range(10):
            for j in range(10):
                points1[i][0] = xx0 + i * dx
                points1[i][1] = yy0 + j * dy
               
        points1, points2 = pai.find_opt_flow(area1, area2, method=method)
        N = len(points1)
        N_ = len(points2)
        print(' N is {} and {}'.format(N, N_))
        out = frame2.copy()
        norms = deque(maxlen=N)
        for i in range(N):
            norms.append(np.linalg.norm(points1[i] - points2[i]))
        mid_norm = sum(norms) / N
        for i in range(N):
            p1 = points1[i] + mask[0]
            p2 = points2[i] + mask[0]
            if np.linalg.norm(p1 - p2) < mid_norm:
                out = draw.draw_point(out, p1, radius=3)
                out = draw.draw_point(out, p2, radius=3)
                out = draw.draw_arrow(out, p1, p2)
        out = draw.draw_text(img=out, pt=(3*x//4, 80), text='points: {}'.format(N),color=draw.blue, font_scale=1, line_type=2)
        out = draw.draw_text(img=out, pt=(0, 80), text='{}'.format(itt),color=draw.blue, font_scale=1, line_type=2)
    #    out = draw.draw_text(out, (3*x//4, 80),  text_properties)
#        small = cv2.resize(out, (0,0), fx=0.7, fy=0.7) 
#        cv2.imshow('frame', small)
        cv2.imwrite('out/{}.jpg'.format(itt), out)
#        #cv2.imshow('area', area)
#        
#        k = cv2.waitKey(20)
        frame1 = frame2
        
        
        tk = time.time()
        times.append(tk - t0)
        T = sum(times) / len(times)
        T = T * (lengh - itt)
        t_min = int(T // 60)
        t_sec = T % 60
        print('{} min {:.02f} sec'.format(t_min, t_sec))
        
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    action('lk')
    vm.action('lk')
