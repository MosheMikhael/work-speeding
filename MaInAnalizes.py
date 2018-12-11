# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:55:16 2018

@author: moshe.f
"""


import cv2
import numpy as np
import collections
import MAIN as mn
import point_at_infinity as pai
import spectrum_saving as spec
import draw
#import service_functions as sf
#import test_courners_finder as cf
#import base_classes as bc
import sys
#import os
import time
def sum_of_residuals(res):
    summ = 0
    N = len(res)
    for i in range(N):
        summ += res[i][0]
    return summ / N

def build_model(pts1, pts2, pai, L):
    n_pts = len(pts1)
                
    F_x = collections.deque(maxlen=n_pts)
    F_y = collections.deque(maxlen=n_pts)
    F_z = collections.deque(maxlen=n_pts)
    F_f = collections.deque(maxlen=n_pts)
    F_o = collections.deque(maxlen=n_pts)
                
    for k in range(n_pts):
        p1 = pts1[k]
        p2 = pts2[k]
        u = p1[0] - pai[0]
        v = p1[1] - pai[1]
        du_x, dv_x = spec.vector_flow_rotation_x(u, v, L)
        du_y, dv_y = spec.vector_flow_rotation_y(u, v, L)
        du_z, dv_z = spec.vector_flow_rotation_z(u, v)
        du_f, dv_f = spec.vector_flow_forward(u, v, L)
        
        F_x.append(np.array([du_x, dv_x], np.float32))
        F_y.append(np.array([du_y, dv_y], np.float32))
        F_z.append(np.array([du_z, dv_z], np.float32))
        F_f.append(np.array([du_f, dv_f], np.float32))
        F_o.append(p2 - p1)
    M = np.zeros((4, 4), np.float32)
    V = np.zeros(4, np.float32)
           
    for k in range(n_pts):
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
    return A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o
        
def get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2):
    residuals = []
    for k in range(len(pts1)):
        vx = np.array([F_x[k][0], F_x[k][1]], np.float32)
        vy = np.array([F_y[k][0], F_y[k][1]], np.float32)
        vz = np.array([F_z[k][0], F_z[k][1]], np.float32)
        vf = np.array([F_f[k][0], F_f[k][1]], np.float32)
        model_vector = A_x * vx + A_y * vy + A_z * vz + B * vf
        obsrv_vector = np.array([F_o[k][0], F_o[k][1]], np.float32)
        resdl_vector = model_vector - obsrv_vector
        norm_res_vector = np.linalg.norm(resdl_vector) / (np.linalg.norm(obsrv_vector) + 1) # +1 for exception DIV BY ZERO
        residuals.append([float(norm_res_vector), pts1[k], pts2[k], model_vector ,resdl_vector]) # why
    return residuals

def action():
    video_adr = 'input/calibration_video.mp4'
    cam = cv2.VideoCapture(video_adr)
    _, img = cam.read()  
    img2 = img
    y, x, z = img.shape
    L = 10_000
    LINES = 20 # 200
    intensity_threshould = 25
    worst = 10 # percents
    NUMBER_OF_FRAMES = 30
    rectungle_border = 150
    pai_in_frame = np.array([x // 2 - 50,  y * 0.4], np.float32)
    index = 0
    coords = []
    step_uv = 5
    step_u = step_uv # 2
    step_v = step_uv # 2
    draw_trigger = False
    
    
    for du in range(-rectungle_border, rectungle_border + 1, step_u):
        for dv in range(-rectungle_border, rectungle_border + 1, step_v):
            coords.append([du, dv])
    num_u = int(2 * rectungle_border / step_u + 1)
    num_v = int(2 * rectungle_border / step_v + 1)

    
    for frame_iterator in range(NUMBER_OF_FRAMES):
        img1 = img2
        img2 = cam.read()[1]        
        data = []
        residuals_best = -1
        pt_best = -1
        sum_best = 100_000_000

        out_arrays, out_resdul = 0, 0
        print(' START {} frame'.format(frame_iterator))
        with open('output/times.txt', 'a') as file:
            file.write(' START {} frame\n'.format(frame_iterator))
        t0_itt = time.time()
        # Find interesting points
        
        all_points, count_of_interesting_points = mn.get_points(550, 40 * np.pi / 180, LINES, pai_in_frame)
        interesting_points = mn.intensity_threshould_filter_of_points(img1, all_points, intensity_threshould)
        # optical flow
        mod_pts = mn.modification_points(interesting_points)
        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
            
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1, pts2, pai_in_frame, L)
        residuals = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2)
        residuals.sort(key=lambda r: r[0]) 

        actual_lenght = int(len(residuals) * (1 - worst / 100))
        pts1_clean = np.zeros((actual_lenght, 2), np.float32)
        pts2_clean = np.zeros((actual_lenght, 2), np.float32)
        for k in range(actual_lenght):
            pt1 = residuals[k][1]
            pt2 = residuals[k][2]
            pts1_clean[k][0] = pt1[0]
            pts1_clean[k][1] = pt1[1]
            pts2_clean[k][0] = pt2[0]
            pts2_clean[k][1] = pt2[1]
        
        if frame_iterator == 1:
            sys.exit('DONE')
        for du in range(-rectungle_border, rectungle_border + 1, step_u):
            for dv in range(-rectungle_border, rectungle_border + 1, step_v):
                if du == 0 and dv == 0:
                    draw_trigger = True
                print('  Analysis point at du={} dv={}'.format(du, dv), end=' > > >')
                with open('output/times.txt', 'a') as file:
                    file.write('  Analysis point at du={} dv={} > > >'.format(du, dv))
                t0_uv = time.time()
                pai_current = pai_in_frame + np.array([du, dv])
                
                A_x_clean, A_y_clean, A_z_clean, B_clean, F_x_clean, F_y_clean, F_z_clean, F_f_clean, F_o_clean = build_model(pts1_clean, pts2_clean, pai_current, L)
                residuals_clean = get_residuals(A_x_clean, A_y_clean, A_z_clean, B_clean, F_x_clean, F_y_clean, F_z_clean, F_f_clean, F_o_clean, pts1_clean, pts2_clean)
                if draw_trigger:
                    out_arrays = draw.draw_point(img2, pai_current, radius=5, thickness=4, color=draw.blue)
                    out_resdul = draw.draw_point(img2, pai_current, radius=5, thickness=4, color=draw.blue)
                    out_arrays = draw.draw_point(out_arrays, pai_in_frame, radius=3, thickness=3, color=draw.dark_red)
                    out_resdul = draw.draw_point(out_resdul, pai_in_frame, radius=3, thickness=3, color=draw.dark_red)
                with open('output/{}_data_{}-du={}dv={}.txt'.format(index, frame_iterator, du, dv), 'a') as file:
                    for k in range(actual_lenght):
                # 0 - |res| / (norm + 1)
                # 1 - current point, x - coord
                # 2 - current point, y - coord
                # 3 - optical flow point, x - coord
                # 4 - optical flow point, y - coord
                # 5 - model vector, x - coord
                # 6 - model vector, y - coord
                # 7 - res vector, x - coord
                # 8 - res vector, y - coord
                        file.write('{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(residuals_clean[k][0], residuals_clean[k][1][0], residuals_clean[k][1][1], residuals_clean[k][2][0], residuals_clean[k][2][1], residuals_clean[k][3][0], residuals_clean[k][3][1], residuals_clean[k][4][0], residuals_clean[k][4][1]))
                        if draw_trigger:
                            mdl_vector = A_x_clean * F_x_clean[k] + A_y_clean * F_y_clean[k] + A_z_clean * F_z_clean[k] + B_clean * F_f_clean[k]
                            out_arrays = draw.draw_arrow(out_arrays, residuals_clean[k][1], residuals_clean[k][1] + mdl_vector, color=draw.cyan) # model
                            out_arrays = draw.draw_arrow(out_arrays, residuals_clean[k][1], residuals_clean[k][2]) # optical flow
                            res = mdl_vector - (residuals_clean[k][2] - residuals_clean[k][1])
                            out_resdul = draw.draw_arrow(out_resdul, residuals_clean[k][1], residuals_clean[k][1] + res, color=draw.red)
                        
                for k in range(actual_lenght, len(residuals)):
                    pt1 = residuals[k][1]
                    pt2 = residuals[k][2]
                    ptm = residuals[k][3]
                    ptr = residuals[k][4]
                    if draw_trigger:
                        out_arrays = draw.draw_arrow(out_arrays, pt1, pt1 + ptm, color=draw.cyan)
                        out_arrays = draw.draw_arrow(out_arrays, pt1, pt2, color=draw.red)
                        out_resdul = draw.draw_arrow(out_resdul, pt1, pt1 + ptr, color=draw.orange)
                if draw_trigger:
                    cv2.imwrite('output/pic/{}_frame_{}-du={}dv={}_optfl&mod.png'.format(index, frame_iterator, du, dv), out_arrays)
                    cv2.imwrite('output/pic/{}_frame_{}-du={}dv={}_residuals.png'.format(index, frame_iterator, du, dv), out_resdul)
                    
                T_uv = time.time() - t0_uv
                print('  {} min {:02.04f} sec  |  id = {}'.format(int(T_uv // 60), T_uv % 60, index))
                with open('output/times.txt', 'a') as file:
                    file.write('  {} min {:02.04f} sec  |  id = {}\n'.format(int(T_uv // 60), T_uv % 60, index))
                sum_current = sum_of_residuals(residuals_clean)
                data.append(sum_current)
                if sum_current < sum_best:
                    pt_best = pai_current
                    residuals_best = residuals
                    sum_best = sum_current
                index += 1
                draw_trigger = False
              
        out_arrays = draw.draw_point(img2, pai_in_frame, radius=5, thickness=3, color=draw.dark_red)
        out_resdul = draw.draw_point(img2, pai_in_frame, radius=5, thickness=3, color=draw.dark_red)
        out_arrays = draw.draw_point(out_arrays, pt_best, radius=3, thickness=4, color=draw.blue)
        out_resdul = draw.draw_point(out_resdul, pt_best, radius=3, thickness=4, color=draw.blue)
        for k in range(len(residuals_best)):
            pt1 = residuals_best[k][1]
            pt2 = residuals_best[k][2]
            ptm = residuals_best[k][3]
            ptr = residuals_best[k][4]
            out_arrays = draw.draw_arrow(out_arrays, pt1, pt1 + ptm, color=draw.cyan)
            out_arrays = draw.draw_arrow(out_arrays, pt1, pt2, color=draw.red)
            out_resdul = draw.draw_arrow(out_resdul, pt1, pt1 + ptr, color=draw.orange)
            
        cv2.imwrite('output/pic/{}_{}_{}_main.png'.format(index, du, dv), out_arrays)
        cv2.imwrite('output/pic/{}_{}_{}_res.png'.format(index, du, dv), out_resdul)

        cv2.imwrite('output/pic/{}.png'.format(frame_iterator), spec.draw_spectrum(x=800,y=800, rows=num_v, cols=num_u, data=data, coords=coords, step_u=step_u, step_v=step_v))
        T_itt = time.time() - t0_itt
        print(' END {} frame at {} min {} sec'.format(frame_iterator, int(T_itt // 60), T_itt % 60))       
                
#        with open('output/times.txt', 'a') as file:
#            file.write(' END {} frame at {} min {:02.04f} sec'.format(frame_iterator, int(T_itt // 60), T_itt % 60))
        
if __name__ == '__main__':
#    sys.exit('DONE')
    action()
    print('DONE')
    
# TO DO:
# 1 - comments
# 2 - variables on the top
# 3 - cheking of points in finding points
# 4 - outliers
# 5 - profiling
    