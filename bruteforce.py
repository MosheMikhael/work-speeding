# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:45:49 2018

@author: moshe.f
"""

import cv2
import draw
import numpy as np
import test_courners_finder as cf
import base_classes as bc
import sys
import time
import collections
import MAIN as mn
import os
import point_at_infinity as pai
import service_functions as sf
import spectrum_saving as spec
#sys.exit('DONE')

def vector_flow_forward(u, v, L, H=1):
    u_ = - u * v / (H * L)
    v_ = - v ** 2 / (H * L)
    return u_, v_
def vector_flow_rotation_x(u, v, L):
    u_ = -v * u / L     
    v_ = L + v ** 2 / L
    return u_, v_

def vector_flow_rotation_y(u, v, L):
    u_ = L + u ** 2 / L
    v_ = u * v / L 
    return u_, v_

def vector_flow_rotation_z(u, v):
    u_ = v
    v_ = -u
    return u_, v_

def do():
    video_adr = 'input/calibration_video.mp4'
    cam = cv2.VideoCapture(video_adr)
    _, img = cam.read()
    y, x, z = img.shape
    x1, y1, x2, y2 = x // 5, 0.45 * y // 1, 4 * x // 5, 0.85 * y // 1
    p1_rec = np.array([x1, y1])
    p2_rec = np.array([x2, y2])
    m_start = np.array([x // 2 - 50,  y * 0.4], np.float32)
    #mask = [p1_rec, p2_rec]
    L = 10_000
    LINES = 250
    intensity_threshould = 25
    rows = 10
    cols = 20
    weight = 0.1
    N = rows * cols
    arrow_size_factor = 10
    
    max_cos = np.cos(20 * np.pi / 180)
    LINES = 20
    img2 = img
    grid = cf.get_grid(p1_rec, p2_rec, rows, cols)
    for itt_frame in range(4):
        img1 = img2
        img2 = cam.read()[1]
        out = draw.draw_rectangle(img2, [m_start + np.array([-5, -5], np.float32), m_start + np.array([5, 5], np.float32)], color=draw.cyan)
        results = []
        time_per_frame = 0   
        for du_m in range(-5, 5 + 1):
            for dv_m in range(-5, 5 + 1):
                T = 0
                m_temp = m_start + np.array([du_m, dv_m], np.float32)
                points, count = mn.get_points(550, 30 * np.pi / 180, LINES, m_temp)
            
                F_x = collections.deque()
                F_y = collections.deque()
                F_z = collections.deque()
                F_f = collections.deque()
                F_o = collections.deque()
                vector_result = []
#                residuals = collections.deque()
#                POINTS = collections.deque()
                print('START {} ITERATION'.format(itt_frame))
                if not _:
                    break
                print(' START points generation:', end=' ')
                t0 = time.time()
                pts_filtered = mn.intensity_threshould_filter_of_points(img1, points, intensity_threshould)
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                print(' START optical flow:', end=' ')
                t0 = time.time()
                mod_pts = mn.modification_points(pts_filtered)
                pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                print(' START buildinig of vectors:', end=' ')
                t0 = time.time()            
                for i in range(len(pts1)):
                    pt1 = pts1[i]
                    pt2_o = pts2[i]
#                    POINTS.append(pts1[i])
                    u = pts1[i][0] - m_temp[0]
                    v = pts1[i][1] - m_temp[1]
                    du_x, dv_x = vector_flow_rotation_x(u, v, L)
                    du_y, dv_y = vector_flow_rotation_y(u, v, L)
                    du_z, dv_z = vector_flow_rotation_z(u, v)
                    du_f, dv_f = vector_flow_forward(u, v, L)
                    
                    F_x.append([du_x, dv_x])
                    F_y.append([du_y, dv_y])
                    F_z.append([du_z, dv_z])
                    F_f.append([du_f, dv_f])
                    F_o.append(pts2[i] - pts1[i])
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                print(' START calculations:', end=' ')
                t0 = time.time()
                M = np.zeros((4, 4), np.float32)
                V = np.zeros(4, np.float32)
           
                for i in range(len(F_f)):
                    M[0][0] += F_x[i][0] ** 2 + F_x[i][1] ** 2
                    M[0][1] += F_x[i][0] * F_y[i][0] + F_x[i][1] * F_y[i][1]
                    M[0][2] += F_x[i][0] * F_z[i][0] + F_x[i][1] * F_z[i][1]
                    M[0][3] += F_x[i][0] * F_f[i][0] + F_x[i][1] * F_f[i][1]
                    M[1][1] += F_y[i][0] ** 2 + F_y[i][1] ** 2
                    M[1][2] += F_y[i][0] * F_z[i][0] + F_y[i][1] * F_z[i][1]
                    M[1][3] += F_y[i][0] * F_f[i][0] + F_y[i][1] * F_f[i][1]
                    M[2][2] += F_z[i][0] ** 2 + F_z[i][1] ** 2
                    M[2][3] += F_z[i][0] * F_f[i][0] + F_z[i][1] * F_f[i][1]
                    M[3][3] += F_f[i][0] ** 2 + F_f[i][1] ** 2
                
                    V[0] += F_o[i][0] * F_x[i][0] + F_o[i][1] * F_x[i][1]
                    V[1] += F_o[i][0] * F_y[i][0] + F_o[i][1] * F_y[i][1]
                    V[2] += F_o[i][0] * F_z[i][0] + F_o[i][1] * F_z[i][1]
                    V[3] += F_o[i][0] * F_f[i][0] + F_o[i][1] * F_f[i][1]
                    for i in range(4):
                        for j in range(i):
                            M[i][j] = M[j][i]
                A_x, A_y, A_z, B = np.linalg.solve(M, V)   
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                print(' START drawing:', end=' ')
                t0 = time.time()
                summ = 0
                counter = 0
                for i in range(len(pts1)):
                    pt = pts1[i]
                    vvx = A_x * np.array(F_x[i], np.float32)
                    vvy = A_y * np.array(F_y[i], np.float32)
                    vvz = A_z * np.array(F_z[i], np.float32)
                    vvf = B * np.array(F_f[i], np.float32)
                    pt2_v = vvx + vvy + vvz + vvf            
                    pt2_o = F_o[i]
                    r = pt2_o - pt2_v
                    norm_r = np.linalg.norm(r)
                    vector_result.append(norm_r)
                    summ += norm_r
                    counter += 1            
#                    with open('out/rotations/{}_{}/data.txt'.format(du_m, dv_m), 'a') as f:
#                        f.write('{}  /  {}  =  {} | summ = {}\n'.format(summ, counter, summ / counter, summ))            
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                t0 = time.time()
                print(' Modification of points of infinity:', end=' ')
                tk = time.time()
                print(tk - t0)
                T += tk - t0
                print('TIME: {} min {:02.02f} sec'.format(int((T) // 60), (T) % 60))
                results.append([summ, m_temp])
#                with open('out/rotations/frame{}.txt'.format(itt_frame), 'a') as file:
#                    file.write('{}|{}|{}|{}\n'.format(du_m, dv_m, summ, counter))
                    
                time_per_frame += T
                vector_result.sort()
                LEN = int(0.9 * len(vector_result))
                counter = 0
                summ = 0
                for i in range(LEN):
                    counter += 1
                    summ += vector_result[i]
                with open('out/rotations/frame{}.txt'.format(itt_frame), 'a') as file:
                    file.write('{}|{}|{}|{}\n'.format(du_m, dv_m, summ, counter))
                
        r = min(results)
        print('END FRAME ANALISYS: {} min {:02.04f} sec'.format(int(time_per_frame // 60), time_per_frame % 60))
        out = draw.draw_arrow(out, m_start, r[1])
        cv2.imwrite('out/rotations/{}.jpg'.format(itt_frame), out)
        m_start = (1 - weight) * m_start + weight * r[1]
print('DONE')      

if __name__ == '__main__':
    do()          
    spec.do()