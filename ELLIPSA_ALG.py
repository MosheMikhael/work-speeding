# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:52:24 2018

@author: moshe.f
"""

import cv2
import numpy as np
import collections
import MAIN as mn
import point_at_infinity as pai
import spectrum_saving as spec
import draw
import os
import time
import sys

def sum_of_residuals(res):
    '''
        Sum of residuals in residuals structure.
        
    :param res: residuals structure.
    :return: sum of residuals / number of points.
    '''
    summ = 0
    N = len(res)
    for i in range(N):
        summ += res[i][0]
    return summ / N

def build_model(pts1, pts2, pai, L):
    '''
        This function returns parameters of model.
    :param pts1: array of first point in optical flow.
    :param pts2: array of second point in optical flow.
    :param pai: point of infinity.
    :param L: focus lenght.
    :return: sum of residuals / number of points.
    '''
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

def build_model_2(pts1, pts2, pai, L):
    '''
        This function returns parameters of model.
    :param pts1: array of first point in optical flow.
    :param pts2: array of second point in optical flow.
    :param pai: point of infinity.
    :param L: focus lenght.
    :return: sum of residuals / number of points.
    '''
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
    M = np.zeros((3, 3), np.float32)
    V = np.zeros(3, np.float32)
           
    for k in range(n_pts):
        M[0][0] += F_x[k][0] ** 2 + F_x[k][1] ** 2
        M[0][1] += F_x[k][0] * F_y[k][0] + F_x[k][1] * F_y[k][1]
        M[0][2] += F_x[k][0] * F_z[k][0] + F_x[k][1] * F_z[k][1]
#        M[0][3] += F_x[k][0] * F_f[k][0] + F_x[k][1] * F_f[k][1]
        M[1][1] += F_y[k][0] ** 2 + F_y[k][1] ** 2
        M[1][2] += F_y[k][0] * F_z[k][0] + F_y[k][1] * F_z[k][1]
#        M[1][3] += F_y[k][0] * F_f[k][0] + F_y[k][1] * F_f[k][1]
        M[2][2] += F_z[k][0] ** 2 + F_z[k][1] ** 2
#        M[2][3] += F_z[k][0] * F_f[k][0] + F_z[k][1] * F_f[k][1]
#        M[3][3] += F_f[k][0] ** 2 + F_f[k][1] ** 2
        
        V[0] += F_o[k][0] * F_x[k][0] + F_o[k][1] * F_x[k][1]
        V[1] += F_o[k][0] * F_y[k][0] + F_o[k][1] * F_y[k][1]
        V[2] += F_o[k][0] * F_z[k][0] + F_o[k][1] * F_z[k][1]
#        V[3] += F_o[k][0] * F_f[k][0] + F_o[k][1] * F_f[k][1]
    for k in range(3):
        for l in range(k):
            M[k][l] = M[l][k]
    A_x, A_y, A_z = np.linalg.solve(M, V)  
    return A_x, A_y, A_z
        
def get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2):
    '''
        Generation a residual structure that contains for each point:
            0 - value of residual
            1 - coordinate of a current point
            2 - coordinate of optical flaw
            3 - model vector
            4 - residual vector
    :param A_x: coefficient rotation-x in model.
    :param A_y: coefficient rotation-y in model.
    :param A_z: coefficient rotation-z in model.
    :param B:   coefficient - in model.
    :param F_x: array of vectors of x-rotations.
    :param F_y: array of vectors of y-rotations.
    :param F_z: array of vectors of z-rotations.
    :param F_f: array of vectors of -rotations.
    :param F_o: array of vectors of observes.
    :param pts1: array of first point in optical flow.
    :param pts2: array of second point in optical flow.
    :return: sum of residuals / number of points.
    '''
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
        residuals.append([norm_res_vector, pts1[k], pts2[k], model_vector ,resdl_vector]) # why
    return residuals

def del_last_files():
    '''
        Service function for removing ald output of script.
        
    :return: None.
    '''
    dlist = os.listdir('output')
    for f in dlist:
        if f.split('.')[-1] == 'png':
            os.remove('output/' + f)

def time_analisys(t1, strr = ''):
    t2 = time.time()
    T = t2 - t1
#    t_hor = int(T // 3600)
#    t_min = int(T // 60)
    t_sec = T % 60
    print('\t' + strr + 'TIME: {:02.010f} sec'.format(t_sec))
    return t_sec

def make_txt(cam, dyn_points10, dyn_points25, dyn_points50, dyn_points75, dyn_points100):
    img = cam.read()[1]
    pt10_e = dyn_points10[0][3]
    pt25_e = dyn_points25[0][3]
    pt50_e = dyn_points50[0][3]
    pt75_e = dyn_points75[0][3]
    pt100_e = dyn_points100[0][3]
    for i in range(len(dyn_points10)):
        pt10_s = dyn_points10[i][3]
        pt25_s = dyn_points25[i][3]
        pt50_s = dyn_points50[i][3]
        pt75_s = dyn_points75[i][3]
        pt100_s = dyn_points100[i][3]
        img = draw.draw_arrow(img, pt10_e, pt10_s, color=draw.green, thickness=3)
        img = draw.draw_arrow(img, pt25_e, pt25_s, color=draw.blue, thickness=2)
        img = draw.draw_arrow(img, pt50_e, pt50_s, color=draw.red, thickness=1)
        img = draw.draw_arrow(img, pt75_e, pt75_s, color=draw.purple, thickness=1)
        img = draw.draw_arrow(img, pt100_e, pt100_s, color=draw.gold, thickness=1)
        pt10_e = pt10_s
        pt25_e = pt25_s
        pt50_e = pt50_s
        pt75_e = pt75_s
        pt100_e = pt100_s
    cv2.imwrite('output/_.png', img)
    with open('output/out.txt', 'w') as file:
        for i in range(len(dyn_points10)):
            line = 'frame {}:\n'.format(i)
            file.write(line)
            line = ' 10: [{:e}, {:e}, {:e}, {:e}], {:e}, {}, {}, {}\n'.format(dyn_points10[i][0][0],dyn_points10[i][0][1],dyn_points10[i][0][2],dyn_points10[i][0][3], dyn_points10[i][1], dyn_points10[i][2], dyn_points10[i][3][0], dyn_points10[i][3][1])
            file.write(line)
            line = ' 25: [{:e}, {:e}, {:e}, {:e}], {:e}, {}, {}, {}\n'.format(dyn_points25[i][0][0],dyn_points25[i][0][1],dyn_points25[i][0][2],dyn_points25[i][0][3], dyn_points25[i][1], dyn_points25[i][2], dyn_points25[i][3][0], dyn_points25[i][3][1])
            file.write(line)
            line = ' 50: [{:e}, {:e}, {:e}, {:e}], {:e}, {}, {}, {}\n'.format(dyn_points50[i][0][0],dyn_points50[i][0][1],dyn_points50[i][0][2],dyn_points50[i][0][3], dyn_points50[i][1], dyn_points50[i][2], dyn_points50[i][3][0], dyn_points50[i][3][1])
            file.write(line)
            line = ' 75: [{:e}, {:e}, {:e}, {:e}], {:e}, {}, {}, {}\n'.format(dyn_points75[i][0][0],dyn_points75[i][0][1],dyn_points75[i][0][2],dyn_points75[i][0][3], dyn_points75[i][1], dyn_points75[i][2], dyn_points75[i][3][0], dyn_points75[i][3][1])
            file.write(line)
            line = '100: [{:e}, {:e}, {:e}, {:e}], {:e}, {}, {}, {}\n'.format(dyn_points100[i][0][0],dyn_points100[i][0][1],dyn_points100[i][0][2],dyn_points100[i][0][3], dyn_points100[i][1], dyn_points100[i][2], dyn_points100[i][3][0], dyn_points100[i][3][1])
            file.write(line)
    T10, T25, T50, T75, T100 = 0, 0, 0, 0, 0
    for i in range(len(dyn_points10)):
        T10 += dyn_points10[i][2]
        T25 += dyn_points25[i][2]
        T50 += dyn_points50[i][2]
        T75 += dyn_points75[i][2]
        T100 += dyn_points100[i][2]
    with open('output/out.txt', 'a') as file:
        line = 'TOTAL:\n'
        file.write(line)
        line = ' 10: {:02.05f} sec\n'.format(T10)
        file.write(line)
        line = ' 25: {:02.05f} sec\n'.format(T25)
        file.write(line)
        line = ' 50: {:02.05f} sec\n'.format(T50)
        file.write(line)
        line = ' 75: {:02.05f} sec\n'.format(T75)
        file.write(line)
        line = '100: {:02.05f} sec\n'.format(T100)
        file.write(line)
    
def build_matrix(pai_in_frame, delta_vector, pts1_clean, pts2_clean, L):
    sum_current = np.zeros(5, np.float32)
    M = np.zeros((5, 5), np.float32)
    for i in range(5):
        pt_current = pai_in_frame + delta_vector[i] # current point for analysis
#        t_ = time.time()
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1_clean, pts2_clean, pt_current, L) # current model
#        cef_current.append([A_x, A_y, A_z, B])
#        time_analisys(t_, 'building model: ')
#        t_ = time.time()
        residuals_clean = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # current residuals
#        time_analisys(t_, 'building residuals: ')
#        t_ = time.time()
        sum_current[i] = sum_of_residuals(residuals_clean) 
        
#        time_analisys(t_, 'summs( {:04d} ): '.format(int(len(pts1_clean))))
        x_ = pt_current[0]
        y_ = pt_current[1]
        M[i][0] = x_  ** 2
        M[i][1] = x_
        M[i][2] = y_  ** 2
        M[i][3] = y_
        M[i][4] = 1
#    time_analisys(t, 'matrix generating: ')
#    t = time.time()
    return M, sum_current, residuals_clean, A_x, A_y, A_z, B
#sys.exit()
# START SCRIPT
del_last_files()

video_adr = 'input/calibration_video.mp4'
cam = cv2.VideoCapture(video_adr)
_, img2 = cam.read()  
y, x, z = img2.shape
pai_in_frame = np.array([x // 2 - 50,  y * 0.4], np.float32)

# MAGIC NUMBERS
intensity_threshould = 25 # used in generating interesting points
worst = 10 # percents
L = 10_000 # focus lenght
LINES = [10, 25, 50, 75, 100] # number of axis from point of infinity for generating interesting points
dyn_points = [collections.deque(), collections.deque(), collections.deque(), collections.deque(), collections.deque()]

temp = -1
par_change = 0.01
eps = -1
best_model = -1
step_uv = 10
step_u = step_uv # 2
step_v = step_uv # 2


debug = False # debug parameter
len_max = 500_000
NUMBER_OF_FRAMES = 30       # Number of frames that will be analysed
NUMBER_OF_STEPS = 4          # Number of steps for each frame
MIN_MODEL_VECTOR = 2         # Min norma of interesting model vector 
COEFF_OPT_MORETHAN_MODEL = 2 # Interesting points: norma of model vector > COEFF.. * norma of optical flaw vector


delta_vector = np.zeros((5, 2), np.float32)
delta_vector[0][0] = 0
delta_vector[0][1] = -step_v
delta_vector[1][0] = -step_u
delta_vector[1][1] = 0
delta_vector[2][0] = 0
delta_vector[2][1] = 0
delta_vector[3][0] = step_u
delta_vector[3][1] = 0

delta_vector[4][0] = 0
delta_vector[4][1] = step_v

all_points = []

for k in range(5):
    all_points_, count_of_interesting_points = mn.get_points(550, 40 * np.pi / 180, LINES[k], pai_in_frame) # generating vector of all points
    all_points.append(all_points_)
    
for frame_iterator in range(NUMBER_OF_FRAMES): # main loop (going from frame to frame)
    print('START FRAME: {}'.format(frame_iterator))
    img1 = img2
    img2 = cam.read()[1] # reading new frame
    out = img2.copy()
#    t = time.time()
#    t_frame = time.time()
    interesting_points = []
    for k in range(5):
        print(' {} analyse'.format(LINES[k]))
        interesting_points_ = mn.intensity_threshould_filter_of_points(img1, all_points[k], intensity_threshould) # generation an interesting points for current frame
        interesting_points.append(interesting_points_)
    # optical flow
        mod_pts = mn.modification_points(interesting_points[k])
        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
    
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1, pts2, pai_in_frame, L) # preliminary model
        residuals = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
    
    # drawing
        if debug:
            for r_i in range(min(len_max, len(residuals))):
                r = residuals[r_i]
                pt1_f = r[1]
                pt2_f = r[2]
                pt1_m = pt1_f
                pt2_m = r[3] + pt1_f
                out = draw.draw_arrow(out, pt1_f, pt2_f, color=draw.red) 
                out = draw.draw_arrow(out, pt1_m, pt2_m, color=draw.cyan) 
        
        pts1_clean = collections.deque() # actual points (1 point in optical flaw)
        pts2_clean = collections.deque() # actual points (2 point in optical flaw)
        
        for r_i in range(len(residuals)):
            r = residuals[r_i]
            pt1 = r[1]     # coordinate of first  point of optical flaw
            pt2_opt = r[2] # coordinate of second point of optical flaw
            opt_vector = r[2] - r[1] # coordinate of vector of optical flaw
            model_vector = r[3]      # coordinate of model vector
            rsdl_vector = r[4]       # coordinate of residual vector
            # criterion of interesting points
            if np.linalg.norm(model_vector) > MIN_MODEL_VECTOR and np.linalg.norm(model_vector) < COEFF_OPT_MORETHAN_MODEL * np.linalg.norm(opt_vector):
                pts1_clean.append(pt1)
                pts2_clean.append(pt2_opt)
                if debug:
                    out = draw.draw_arrow(out, pt1, pt2_opt) # drawing
        if debug:
            out = draw.draw_point(out, pai_in_frame, thickness=3, color=draw.blue)
            cv2.imwrite('output/{}_model1.png'.format(frame_iterator), out)
            out = img2.copy()
            
        sum_current = [] # vector of sums of residuals per each points around point of start
        cef_current = []
#        time_analisys(t, 'preanalisys of frame: ')
#        t_steps = time.time()
        for step in range(NUMBER_OF_STEPS):
            print('  START {} STEP'.format(step))
#            t = time.time()
        
            M, sum_current_, residuals_clean, A_x, A_y, A_z, B = build_matrix(pai_in_frame, delta_vector, pts1_clean, pts2_clean, L)
            cef_current.append([A_x, A_y, A_z, B])
            sum_current.append(sum_current_)
            A, A_, B, B_, C_ = np.linalg.solve(M, sum_current_) 
            
#        time_analisys(t, 'solving: ')
#        t = time.time()
#        print('  result:', A, A_, B, B_, C_)
            if step == 0 and debug: # drawing
                for r in residuals_clean:
                        pt1_f = r[1]
                        pt2_f = r[2]
                        pt1_m = pt1_f
                        pt2_m = r[3] + pt1_f
                        out = draw.draw_arrow(out, pt1_f, pt2_f, color=draw.green) 
                        out = draw.draw_arrow(out, pt1_m, pt2_m, color=draw.cyan) 
#        time_analisys(t, ' point of inf modification: ')
            x_new = - A_ / (2 * A) # x - coordinate of new point
            y_new = - B_ / (2 * B) # y - coordinate of new point
            if (x_new - pai_in_frame[0]) ** 2 + (y_new - pai_in_frame[1]) ** 2 > step_uv ** 2: # if new point is out of range
                x_new = pai_in_frame[0] + delta_vector[sum_current[k].tolist().index(min(sum_current[k]))][0]
                y_new = pai_in_frame[1] + delta_vector[sum_current[k].tolist().index(min(sum_current[k]))][1]
        
            
#            print ('out of range')        
        pt = np.array([x_new, y_new], np.float32)
        best_model = cef_current[sum_current[k].tolist().index(min(sum_current[k]))]
        
        #ERROR 
        A_x, A_y, A_z = build_model_2(pts1_clean, pts2_clean, pt, L)
        B = best_model[3]
        residuals_eps = get_residuals(A_x, A_y, A_z, B * (1 + par_change), F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean)
        eps = par_change * B / np.sqrt(sum_of_residuals(residuals_eps) - min(sum_current))
        eps = np.sqrt(eps ** 2)
        pai_in_frame = pt
        if debug:
            out = draw.draw_arrow(out, pai_in_frame, pt, color=draw.orange)
            out = draw.draw_point(out, pt, color=draw.blue, radius=1)
    if debug:
        out = draw.draw_point(out, pai_in_frame, thickness=3, radius=3,color=draw.red)    
        cv2.imwrite('output/{}_model2.png'.format(frame_iterator), out)
            
            
#    time_analisys(t_steps, 'per all steps: ')
#    T = time_analisys(t_frame, 'per frame: ')
    T = -1
    dyn_points[k].append([best_model, eps, T, pai_in_frame])
    out = draw.draw_text(out, (500, 0), 'Hello, world')
    out = draw.draw_text(out, (550, 0), 'A_x = {:02.09f}'.format(best_model[0]))
    out = draw.draw_text(out, (600, 0), 'A_y = {:02.09f}'.format(best_model[0]))
    out = draw.draw_text(out, (650, 0), 'A_z = {:02.09f}'.format(best_model[0]))
    out = draw.draw_text(out, (700, 0), '  B = {:02.09f} +/- {:02.09f}'.format(best_model[0], eps))
        
    cv2.imwrite('output/{}_{}.png'.format(frame_iterator, LINES[k]), out)
#sys.exit('DONE')
#make_txt(cam, dyn_points10, dyn_points25, dyn_points50, dyn_points75, dyn_points100)
print('DONE')

    
