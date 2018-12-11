# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:27:05 2018

@author: moshe.f
"""

import cv2
import os
import time
import collections

def action(name):
    path = []
    itt = 0
    N0 = 1360
    N = 8942
    num = N + 1
    times = collections.deque(maxlen=N)
    for i in range(N0, num):
        path.append('output/{:05d}.png'.format(i))
    
    f = cv2.imread(path[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    y, x, z = f.shape
    out = cv2.VideoWriter(name + '.avi', fourcc, 20.0, (x, y))
    out.write(f)
    while itt < len(path):
        t0 = time.time()
        if os.path.exists(path[itt]):
            f = cv2.imread(path[itt])    
            out.write(f)
            itt += 1
        tk = time.time()
        times.append(tk - t0)
        T = (sum(times) / len(times)) * (num - itt)
        t_min = int(T // 60)
        t_sec = T % 60
        print('{:.02f} % | {}min {:.02f}sec'.format(100 * (itt) / num, t_min, t_sec))
    out.release()
    print('done')

if __name__ == '__main__':
    action('result')