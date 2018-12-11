# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:55:13 2018

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
import 
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

debug = False
video_adr = 'input/calibration_video.mp4'
cam = cv2.VideoCapture(video_adr)
_, img2 = cam.read()
y, x, z = img2.shape
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
grid = cf.get_grid(p1_rec, p2_rec, rows, cols)


#for I in range(700):
#m_points = bc.Points()

for du_m in range(-5, 5 + 1, 1):
    for dv_m in range(-5, 5 + 1, 1):
        
        cam = cv2.VideoCapture(video_adr)   
        img2 = cam.read()[1]
        
        os.mkdir('out/rotations/{}_{}'.format(du_m, dv_m))
        m = m_start + np.array([du_m, dv_m], np.float32)
        points, count = mn.get_points(550, 30 * np.pi / 180, LINES, m)
        for I in range(200):
            T = 0
            img1 = img2
            _, img2 = cam.read()    
        #        t0 = time.time()
            F_x = collections.deque()
            F_y = collections.deque()
            F_z = collections.deque()
            F_f = collections.deque()
            F_o = collections.deque()
            residuals = collections.deque()
            POINTS = collections.deque()
            
            print('START {} ITERATION'.format(I))
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
        #    pts1, pts2 = good_rotation(pts1_, pts2_, m, 30 * np.pi / 180)
        #    img_x = draw.draw_grid(img1, grid)
        #    img_y = draw.draw_grid(img1, grid)
        #    img_z = draw.draw_grid(img1, grid)
        #    img_f = draw.draw_grid(img1, grid)
        #        img_x = draw.draw_point(img1, m, color=draw.blue, radius=5, thickness=3)
        #        img_y = draw.draw_point(img1, m, color=draw.blue, radius=5, thickness=3)
        #        img_z = draw.draw_point(img1, m, color=draw.blue, radius=5, thickness=3)
        #        img_f = draw.draw_point(img1, m, color=draw.blue, radius=5, thickness=3)
            print(' START buildinig of vectors:', end=' ')
            t0 = time.time()
        #        img_o = draw.draw_point(img1, m, color=draw.blue, radius=5, thickness=3)
            
            for i in range(len(pts1)):
        #            ind = i + cols * j
        #            arr += [ind]
        #            print(ind, i, j)
        #            pt1 = (grid[i][j][0] + grid[i][j][1]) / 2
                    
                    pt1 = pts1[i]
                    pt2_o = pts2[i]
                    POINTS.append(pts1[i])
                    u = pts1[i][0] - m[0]
                    v = pts1[i][1] - m[1]
                    du_x, dv_x = vector_flow_rotation_x(u, v, L)
                    du_y, dv_y = vector_flow_rotation_y(u, v, L)
                    du_z, dv_z = vector_flow_rotation_z(u, v)
                    du_f, dv_f = vector_flow_forward(u, v, L)
                    
                    F_x.append([du_x, dv_x])
                    F_y.append([du_y, dv_y])
                    F_z.append([du_z, dv_z])
                    F_f.append([du_f, dv_f])
                    F_o.append(pts2[i] - pts1[i])
        #                du_x /= 10
        #                dv_x /= 10
        #                du_y /= 10
        #                dv_y /= 10
        #                du_z /= 10
        #                dv_z /= 10
        #            du_f /= 10
        #            dv_f /= 10
        #             F_y.append(np.sqrt(du_y ** 2 + dv_y ** 2))
        #            F_z.append(np.sqrt(du_z ** 2 + dv_z ** 2))
        #            F_frd
        #            F_o.append(np.sqrt(du_x ** 2 + dv_x ** 2))
        #                pt2_x = pt1 + np.array([du_x, dv_x], np.float32)
        #                pt2_y = pt1 + np.array([du_y, dv_y], np.float32)
        #                pt2_z = pt1 + np.array([du_z, dv_z], np.float32)
        #                pt2_f = pt1 + np.array([du_f, dv_f], np.float32)
        #                img_x = draw.draw_arrow(img_x, pt1, pt2_x, color=draw.blue)
        #                img_y = draw.draw_arrow(img_y, pt1, pt2_y, color=draw.blue)
        #                img_z = draw.draw_arrow(img_z, pt1, pt2_z, color=draw.blue)
        #                img_f = draw.draw_arrow(img_f, pt1, pt2_f, color=draw.blue)
        
        #            img_o = draw.draw_arrow(img_o, pt1, pt2_x, color=draw.blue)
        #            img_o = draw.draw_arrow(img_o, pt1, pt2_y, color=draw.blue)
        #            img_o = draw.draw_arrow(img_o, pt1, pt2_z, color=draw.blue)
        #            img_o = draw.draw_arrow(img_o, pt1, pt2_f, color=draw.blue)
        #                img_o = draw.draw_arrow(img_o, pt1, pt2_o, color=draw.green)
        #                img_o = draw.draw_arrow(img_o, pt1, pt1 + F_o[i], color=draw.red)
        #    cv2.imwrite('out/rotations/{}_img_x.jpg'.format(I), img_x)
        #    cv2.imwrite('out/rotations/{}_img_y.jpg'.format(I), img_y)
        #    cv2.imwrite('out/rotations/{}_img_z.jpg'.format(I), img_z)
        #    cv2.imwrite('out/rotations/{}_img_f.jpg'.format(I), img_f)
        #    cv2.imwrite('out/rotations/{}_img_o.jpg'.format(I), img_o)
            tk = time.time()
            print(tk - t0)
            T += tk - t0
            
            
            print(' START calculations:', end=' ')
            t0 = time.time()
            M = np.zeros((4, 4), np.float32)
            V = np.zeros(4, np.float32)
            
            m_00 = 0
            m_01 = 0
            m_02 = 0
            m_03 = 0
            m_11 = 0
            m_12 = 0
            m_13 = 0
            m_22 = 0
            m_23 = 0
            m_33 = 0
            
            v_0 = 0
            v_1 = 0
            v_2 = 0
            v_3 = 0
            
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
            
        #        for i in range(4):
        #            for j in range(4):
        #                print(M[i][j], end=' ')
        #            print('| ' + str(V[i]))
            tk = time.time()
            print(tk - t0)
            T += tk - t0
        #        print('result: Ax = {}\n        Ay = {}\n        Az = {}\n         B = {}'.format(A_x, A_y, A_z, B))
        
        # DRAWING
            print(' START drawing:', end=' ')
            t0 = time.time()
    #        img_out = draw.draw_point(img1, m, 5, draw.blue, 4)
    #        img_r = draw.draw_point(img1, m, 5, draw.blue, 4)
        ##        img_x = draw.draw_point(img1, m, 5, draw.blue, 4)
        ##        img_y = draw.draw_point(img1, m, 5, draw.blue, 4)
        ##        img_z = draw.draw_point(img1, m, 5, draw.blue, 4)
        ##        img_f = draw.draw_point(img1, m, 5, draw.blue, 4)
            
            summ = 0
            counter = 0
            for i in range(len(POINTS)):
                pt = POINTS[i]
                vvx = A_x * np.array(F_x[i], np.float32)
                vvy = A_y * np.array(F_y[i], np.float32)
                vvz = A_z * np.array(F_z[i], np.float32)
                vvf = B * np.array(F_f[i], np.float32)
        ##            
        ##    #        V_u = A_x * F_x[i][0] + A_y * F_y[i][0] + A_z * F_z[i][0] + B * F_f[i][0]
        ##    #        V_v = A_x * F_x[i][1] + A_y * F_y[i][1] + A_z * F_z[i][1] + B * F_f[i][1]
        ##    #        pt2_v = np.array([V_u, V_v], np.float32)
                pt2_v = vvx + vvy + vvz + vvf
        ##            
                pt2_o = F_o[i]
                r = pt2_o - pt2_v
                norm_r = np.linalg.norm(r)
                summ += norm_r
                counter += 1
        ##            if debug:
        ##                img_x = draw.draw_arrow(img_x, pt, pt + 10 * vvx)
        ##                img_y = draw.draw_arrow(img_y, pt, pt + 10 * vvy)
        ##                img_z = draw.draw_arrow(img_z, pt, pt + 10 * vvz)
        ##                img_f = draw.draw_arrow(img_f, pt, pt + 10 * vvf)
    #            img_out = draw.draw_arrow(img_out, pt, pt + arrow_size_factor * pt2_o, color = draw.green)
    #            img_out = draw.draw_arrow(img_out, pt, pt + arrow_size_factor * pt2_v, color = draw.cyan)
    #            img_r = draw.draw_arrow(img_r, pt, pt + arrow_size_factor * r, color = draw.red)
    #    #    
    #        img_out = draw.draw_text(img_out, (4 * x / 5 // 1, 50), ' Ax = ' + str(A_x), font_scale=1)
    #        img_out = draw.draw_text(img_out, (4 * x / 5 // 1, 80), ' Ay = ' + str(A_y), font_scale=1)
    #        img_out = draw.draw_text(img_out, (4 * x / 5 // 1, 110), ' Az = ' + str(A_z), font_scale=1)
    #        img_out = draw.draw_text(img_out, (4 * x / 5 // 1, 140), '  B = ' + str(B), font_scale=1)
    #        img_out = draw.draw_text(img_out, (4 * x / 5 // 1, 160), 'sum = ' + str(summ), font_scale=1, line_type=3)
    #        cv2.imwrite('out/rotations/{}/{}_obs.jpg'.format(L, I), img_out)
    #        cv2.imwrite('out/rotations/{}/{}_r.jpg'.format(L, I), img_r)
            
            with open('out/rotations/{}_{}/data.txt'.format(du_m, dv_m), 'a') as f:
                f.write('{}  /  {}  =  {} | summ = {}\n'.format(summ, counter, summ / counter, summ))
        #        if debug:
        #            cv2.imwrite('out/rotations/{}/{}_x.jpg'.format(I, L), img_x)
        #            cv2.imwrite('out/rotations/{}/{}_y.jpg'.format(I, L), img_y)
        #            cv2.imwrite('out/rotations/{}/{}_z.jpg'.format(I, L), img_z)
        #            cv2.imwrite('out/rotations/{}/{}_f.jpg'.format(I, L), img_f)
            
            tk = time.time()
            print(tk - t0)
            T += tk - t0
            t0 = time.time()
            print(' Modification of points of infinity:', end=' ')
#            m_points.add_points(pts1, pts2)
#            m_points.mark_inlier_all()
#            m_points.make_lines()
#            pt_m, st = m_points.get_OF_by_max_dens()
#            if st:
#                m = weight * pt_m + (1 - weight) * m
            tk = time.time()
            print(tk - t0)
            T += tk - t0
            print('TIME: {} min {:02.02f} sec'.format(int((T) // 60), (T) % 60))
    #    img_x = cv2.resize(img_x, (0, 0), fx = 0.5, fy = 0.5)
    #    img_y = cv2.resize(img_y, (0, 0), fx = 0.5, fy = 0.5)
    #    img_z = cv2.resize(img_z, (0, 0), fx = 0.5, fy = 0.5)
    
    #    cv2.imshow('img_x' ,img_x)
    #    cv2.imshow('img_y' ,img_y)
    #    cv2.imshow('img_z' ,img_z)
    #    if cv2.waitKey(50)  == ord('q'):
    #        break
    #cv2.destroyAllWindows()
#    X, Y, D = [], [], []
#    i = -1000
#    path = 'out/rotations/{}/data.txt'
#    while i < 10001:
#        X.append(i)
#        with open(path.format(i), 'r') as f:
#            data = collections.deque()
#            for line in f:
#                s = line.split(' ')
#                data.append(np.float32(s[8]))
#            N = len(data)
#            m = sum(data) / N
#            std = 0
#            for dd in data:
#                std += (dd - m) ** 2
#            print('L = {:4d} | m = {:.09f}  |  std = {:.09f}'.format(i, m, np.sqrt(std / (N - 1))))
#            Y.append(m)
#            D.append(std)
#            if i < 1000:
#                i+=100
#            else:
#                i += 250
#            if i == 0:
#                i += 100
    print('DONE')
    
    #path = 'out/rotations/{}/data.txt'
    #for i in range(100, 1100, 100):
    #    with open(path.format(i), 'r') as f:
    #        data = collections.deque()
    #        for line in f:
    #            s = line.split(' ')
    #            data.append(np.float32(s[8]))
    #        N = len(data)
    #        m = sum(data) / N
    #        std = 0
    #        for dd in data:
    #            std += (dd - m) ** 2
    #        print('L = {:4d} | m = {:.06f}  |  std = {:.06f}'.format(i, m, np.sqrt(std / (N - 1))))
    #L =  100 | m = 3.284056  |  std = 3.045173
    #L =  200 | m = 3.116313  |  std = 3.125327
    #L =  300 | m = 2.991115  |  std = 3.098409
    #L =  400 | m = 2.916693  |  std = 3.038718
    #L =  500 | m = 2.876183  |  std = 2.991276
    #L =  600 | m = 2.853865  |  std = 2.959868
    #L =  700 | m = 2.840820  |  std = 2.939262
    #L =  800 | m = 2.832679  |  std = 2.925243
    #L =  900 | m = 2.827304  |  std = 2.915385
#L = 1000 | m = 2.823581  |  std = 2.908232