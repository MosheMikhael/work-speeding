# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:27:03 2018

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

def vector_flow_forward(u, v, L, H=1):
    tmp = v / (H * L)
    return -u * tmp, -v * tmp

def vector_flow_rotation_x(u, v, L):
    tmp = v / L
    return -u * tmp, L + v * tmp

def vector_flow_rotation_y(u, v, L):
    tmp = u / L
    return L + u * tmp, v * tmp

def vector_flow_rotation_z(u, v):
    return -u, v

def draw_spectrum(x, y, rows, cols, data, coords, step_u=1, step_v=1):
    frame = np.zeros((y, x, 3), np.uint8)
    grid = cf.get_grid([0, 0], [x, y], rows, cols) 
    max_val = max(data)
    min_val = min(data)
    id_min_val = data.index(min_val)
    delta = max_val - min_val
    d_gray = 255 / delta
    print(coords)
#    for i in range(rows):
#        for j in range(cols):
    i_min = coords[0][0]
    j_min = coords[0][1]
    for I in range(len(data)):
        coord = coords[I]
        i = int((coord[0] - i_min) / step_u)
        j = int((coord[1] - j_min) / step_v)
        print(I, i, j)
        pt1 = grid[i][j][0]
        pt2 = grid[i][j][1]
        g = int(d_gray * (data[I] - min_val))
        color = (g, g, g)
        if coord[0] == 0 and coord[1] == 0:
#        if I == id_zero:
#            print(I)
            color = draw.purple
        if I == id_min_val:
            color = draw.gold
        frame[pt1[0]:pt2[0], pt1[1]:pt2[1]] = color
    return frame


def do():
    y_ = 800
    x_ = 800
    video_adr = 'input/calibration_video.mp4'
    cam = cv2.VideoCapture(video_adr)
    _, img = cam.read()
    img2 = img
    y, x, z = img.shape
    m = np.array([x // 2 - 50,  y * 0.4], np.float32)
    L = 10
    LINES = 250
    intensity_threshould = 25
    
    arrow_size_factor = 1000
    N = 5
    
    for itt in range(4):
        img1 = img2
        _, img2 = cam.read()
        points, count = mn.get_points(550, 30 * np.pi / 180, LINES, m)
        pts_filtered = mn.intensity_threshould_filter_of_points(img1, points, intensity_threshould)
        mod_pts = mn.modification_points(pts_filtered)
#        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
        
        coords = [] 
        data = []
        with open('out/rotations/frame{}.txt'.format(itt)) as file:
            for line in file:
                dx, dy, summ, count = line.split('|')
                coords.append([int(dx), int(dy)])
                data.append(np.float32(summ) / np.float32(count))                
    #    sys.exit('')
        min_dx = min(coords)[0]
        min_dy = min(coords)[1]
        max_dx = max(coords)[0]
        max_dy = max(coords)[1]
        min_val_coord = coords[data.index(min(data))]
        cols =  max_dx - min_dx + 1
        rows =  max_dy - min_dy + 1
        frame = draw_spectrum(x_, y_, rows, cols, data, coords)
        m_new = m + np.array(min_val_coord, np.float32)
        dm = (m_new - m) / N
        cv2.imwrite('out/rotations/{}.png'.format(itt), frame)
            
        out = draw.draw_point(img2, m, color=draw.dark_green, thickness=5, radius=3)
        points, count = mn.get_points(550, 30 * np.pi / 180, LINES, m)
        pts_filtered = mn.intensity_threshould_filter_of_points(img1, points, intensity_threshould)
        mod_pts = mn.modification_points(pts_filtered)
        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
        for i in range(N):
            print(' step: ', i, N)
            m_ = m + i * dm
            F_x = collections.deque()
            F_y = collections.deque()
            F_z = collections.deque()
            F_f = collections.deque()
            F_o = collections.deque()
            
            for k in range(len(pts1)):
                if k % (len(pts1) // 10) == 0:
                    print('   arrows: ', k, len(pts1))
                pt1 = pts1[k]
                pt2 = pts2[k]
                u = pt1[0] - m_[0]
                v = pt1[1] - m_[1]
                du_x, dv_x = vector_flow_rotation_x(u, v, L)
                du_y, dv_y = vector_flow_rotation_y(u, v, L)
                du_z, dv_z = vector_flow_rotation_z(u, v)
                du_f, dv_f = vector_flow_forward(u, v, L)

                F_x.append([du_x, dv_x])
                F_y.append([du_y, dv_y])
                F_z.append([du_z, dv_z])
                F_f.append([du_f, dv_f])
                F_o.append(pt2 - pt1)
                
                out = draw.draw_arrow(out, pt1, pt2)
            
            M = np.zeros((4, 4), np.float32)
            V = np.zeros(4, np.float32)
            for k in range(len(F_f)):
                M[0][0] += F_x[k][0] ** 2 + F_x[k][1] ** 2
                M[0][1] += F_x[k][0] * F_y[k][0] + F_x[k][1] * F_y[k][1]
                M[0][2] += F_x[k][0] * F_z[k][0] + F_x[k][1] * F_z[k][1]
                M[0][3] += F_x[k][0] * F_f[k][0] + F_x[k][1] * F_f[k][1]
                M[1][1] += F_y[k][0] ** 2 + F_y[k][1] ** 2
                M[1][2] += F_y[k][0] * F_z[k][0] + F_y[k][1] * F_z[k][1]
                M[1][3] += F_y[k][0] * F_f[k][0] + F_y[k][1] * F_f[k][1]
                M[2][2] += F_z[k][0] ** 2 + F_z[k][1] ** 2
                M[2][3] += F_z[k][0] * F_f[k][0] + F_z[k][1] * F_f[k][1]
                M[3][3] += F_f[k][0] ** 2 + F_f[k][1] ** 2
            
                V[0] += F_o[k][0] * F_x[k][0] + F_o[k][1] * F_x[k][1]
                V[1] += F_o[k][0] * F_y[k][0] + F_o[k][1] * F_y[k][1]
                V[2] += F_o[k][0] * F_z[k][0] + F_o[k][1] * F_z[k][1]
                V[3] += F_o[k][0] * F_f[k][0] + F_o[k][1] * F_f[k][1]
                for k in range(4):
                    for l in range(k):
                        M[k][l] = M[l][k]
            A_x, A_y, A_z, B = np.linalg.solve(M, V)
            print(A_x, A_y, A_z, B)
            res = draw.draw_point(img2, m_, color=draw.dark_green, radius=5, thickness=3)
            for k in range(len(pts1)):
                if k % (len(pts1) // 10) == 0:
                    print('   arrows: ', k, len(pts1))
                pt1 = pts1[k]
                vvx = A_x * np.array(F_x[k], np.float32)
                vvy = A_y * np.array(F_y[k], np.float32)
                vvz = A_z * np.array(F_z[k], np.float32)
                vvf =   B * np.array(F_f[k], np.float32)
                dpt2 = vvx + vvy + vvz + vvf
                out = draw.draw_arrow(out, pt1, pt1 + dpt2, color=draw.cyan)
                res = draw.draw_arrow(res, pt1, pt1 + (pts2[k] - pts1[k]) - dpt2, color=draw.red)
            cv2.imwrite('out/rotations/{}_{}.png'.format(itt, i), out)
            cv2.imwrite('out/rotations/{}_{}_r.png'.format(itt, i), res)
        m = m_new
        print(itt, 30)
    print('DONE')
    
if __name__ == '__main__':
    pass