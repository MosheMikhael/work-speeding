# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:02:43 2018

@author: moshe.f
"""

import cv2
import numpy as np
import collections
import MAIN as mn
import point_at_infinity as pai_module
import base_classes as bc_module
import spectrum_saving as spec
import service_functions as sf
import draw
import matplotlib.pyplot as plt
import os
import time
import sys
class Time:
    def __init__(self, h, m, s):
        self.h = h
        self.m = m
        self.s = s
    def get_time(self, sec):
        _s = self.s + sec
        _m = self.m + _s // 60
        _s = int(_s % 60)
        _h = self.h + int(_m // 60)
        _m = int(_m % 60)
        return _h, _m, _s
    def get_sec(self):
        return self.s + 60 * self.m + 3600 * self.h
def get_data_from_parsed_file(filename):
    ind = []
    lon = []
    lat = []
    dist = []
    v = []
    with open(filename, 'r') as file:
        t_prev = -1
        for line in file:
            p = line.split('|')
            tt = p[0].split(':')
            t_current = [int(tt[0]), int(tt[1]), int(tt[2])]
            lat_current = float(p[1])
            lon_current = float(p[2])
            dist_current = float(p[3])
            v_current = 0
            if t_prev != -1:
                t1 = 3600 * t_current[0] + 60 * t_current[1] + t_current[2]
                t2 = 3600 * t_prev[0] + 60 * t_prev[1] + t_prev[2]
                v_current = get_speed(dist_current, t1, t2)
            
            dist.append(dist_current)
            ind.append(t_current)
            lat.append(lat_current)
            lon.append(lon_current)
            v.append(v_current)     
                    
            t_prev = t_current
    return ind, lon, lat, dist, v

def get_speed(dist, time_1, time_2):
    v = float(3600 * dist / (time_1 - time_2))
    return v

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
    if N == 0:
        return -1
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

def build_model_without_B(pts1, pts2, pai, L):
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
        #delete
        norm_res_vector = np.linalg.norm(resdl_vector) / (np.linalg.norm(obsrv_vector) + 1) # +1 for exception DIV BY ZERO
        residuals.append([norm_res_vector, pts1[k], pts2[k], model_vector ,resdl_vector]) # why
    return residuals

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
    return M, sum_current

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def del_last_files():
    '''
        Service function for removing ald output of script.
        
    :return: None.
    '''
    dlist = os.listdir('output')
    for f in dlist:
        if f.split('.')[-1] == 'png':
            os.remove('output/' + f)

del_last_files()
video_adr = 'input/calibration_video.mp4'
video_adr = 'input/1750be37-5499-4c6e-9421-9bb15b277a94.mp4'
cam = cv2.VideoCapture(video_adr)
_, img2 = cam.read()  
y, x, z = img2.shape
pai_in_frame = np.array([x // 2 - 50,  y * 0.4], np.float32)
#pai_in_frame = np.array([900,  500], np.float32)

# MAGIC NUMBERS
intensity_threshould = 25 # used in generating interesting points
worst = 10 # percents
L = 10_000 # focus lenght
#LINES = [10, 25, 50, 75, 100] # number of axis from point of infinity for generating interesting points
LINES = [100]
pai = np.zeros(len(LINES), object)
angle = 40
lenght_of_axis = 550
max_y_for_points = 400
dyn_points = []
for i in range(len(LINES)):
    dyn_points.append(collections.deque())
    pai[i] = pai_in_frame

filename = 'input/parsed-1750be37-5499-4c6e-9421-9bb15b277a94.txt'
times, lon, lat, dist, speeds = get_data_from_parsed_file(filename)

temp = -1
par_change = 0.01
eps = -1
best_model = -1
step_uv = 20
step_u = step_uv # 2
step_v = step_uv # 2


debug = False # debug parameter
len_max = 500_000
NUMBER_OF_FRAMES = 15_000       # Number of frames that will be analysed
NUMBER_OF_STEPS = 5          # Number of steps for each frame
MIN_MODEL_VECTOR = 2         # Min norma of interesting model vector 
COEFF_OPT_MORETHAN_MODEL = 2 # Interesting points: norma of model vector > COEFF.. * norma of optical flaw vector
MAX_RESIDUAL = 1
OUTLIER_THRESHOLD = 0.5
fs = 1


data_out = []
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
filename = 'output/data.txt'
#sys.exit()
file = open(filename, 'w')
#1.1
for IND_LINES in range(len(LINES)):
    lines = LINES[IND_LINES]
    all_points_, count_of_interesting_points = mn.get_points(lenght_of_axis, angle * np.pi / 180, lines, max_y_for_points, pai_in_frame) # generating vector of all points
    all_points.append(all_points_)
speed_g = []
frame_delay = 1360
print('Generating arr of speed')
for i in range(frame_delay, 15000):
    k = i
    t = k / 25 + 50103
#    x.append(i)
    if i % 100 == 0:
        print(i)
    ii = -1
    for o in range(1, len(times)):
        tt = times[o]
        TIME_temp = Time(tt[0], tt[1], tt[2])
        tt = TIME_temp.get_sec()
        dt = tt - t # разница между временем записи с индексом о и текущим времинем.
        if dt <= 0:
            ii = o
    speed_g.append(speeds[ii])
# yolo
classes_adr = 'input/for yolo/classes.txt'
v3_weights_adr = 'input/for yolo/yolov3.weights'
v3_config_adr  = 'input/for yolo/yolov3.cfg'
classes = None
with open(classes_adr, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(config=v3_config_adr, model=v3_weights_adr)
Width = img2.shape[1]
Height = img2.shape[0]

scale = 0.00392
frame_delay = 5125
for frame_itt in range(NUMBER_OF_FRAMES):
    print('FRAME: {}'.format(frame_itt))
    
    file.write('frame: {}\n'.format(frame_itt))
    file.flush()
    img1 = img2
    img2 = cam.read()[1]
    
    if frame_itt < frame_delay:
        continue
    
    for lines_itt in range(len(LINES)):
        print(' Lines: {:02d}'.format(LINES[lines_itt]), end=' -> ')
        pic_name = 'output/{:02d}_lines={:03d}.png'.format(frame_itt, LINES[lines_itt])
        out = img1.copy()
        out = draw.draw_point(out, pai[lines_itt], 5, draw.red, 5)
        out = cv2.ellipse(out, (pai_in_frame[0], pai_in_frame[1]), (lenght_of_axis, lenght_of_axis), 0, angle, 180 - angle, color=draw.cyan)
        _y_ = int(pai_in_frame[1]) + max_y_for_points
        _x1_ = int(x // 4)
        _x2_ = int(3 * x // 4)
        out = cv2.line(out, (_x1_, _y_), (_x2_, _y_), color=draw.cyan)
        #1.2
        interesting_points = mn.intensity_threshould_filter_of_points(img1, all_points[lines_itt], intensity_threshould) # generation an interesting points for current frame
        if len(interesting_points) == 0:
            print('ERROR')
            cv2.imwrite(pic_name, out)
            continue
        #2
        mod_pts = mn.modification_points(interesting_points)
        pts1, pts2 = pai_module.find_opt_flow_lk_with_points(img1, img2, mod_pts)
        if len(pts1) == 0:
            print('ERROR')
            cv2.imwrite(pic_name, out)
            continue
        
        print()
        #5 CLEANING
#        pts1_clean = collections.deque() # actual points (1 point in optical flaw)
#        pts2_clean = collections.deque() # actual points (2 point in optical flaw)
        points_for_pai = bc_module.Points()
        r_summ = []
        last_value = -1
        t0 = time.time()
        blob = cv2.dnn.blobFromImage(img1, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out_data in outs:
            for detection in out_data:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        # go through the detections remaining
        # after nms and draw bounding box
        boxes_out = []
        for ind in indices:
            i_ = ind[0]
            box = boxes[i_]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]   
            img1 = draw.draw_bounding_box(img1, None, COLORS[class_ids[i_]], confidences[i_], round(x), round(y), round(x+w), round(y+h))
            boxes_out.append([[round(x), round(y)], [round(x+w), round(y+w)]])
#            print(classes[class_ids[i_]], 'was drawed border')
        t0 = time.time()
        pts1_tem = []
        pts2_tem = []
        for i in range(len(pts1)):
            pt1 = pts1[i]
            pt2 = pts2[i]
            o = True
            for box in boxes_out:
                box_pt1 = box[0]
                box_pt2 = box[1]
                if box_pt1[0] <= pt1[0] <= box_pt2[0] and box_pt1[1] <= pt1[1] <= box_pt2[1]:
                    o = False
                    break
            if o:
                pts1_tem.append(pt1)
                pts2_tem.append(pt2)
        print('\t', len(pts1) - len(pts1_tem), 'was filtered by objects.')
        pts1 = pts1_tem
        pts2 = pts2_tem
        #3
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1, pts2, pai_in_frame, L) # preliminary model
        #4
        residuals = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
        B_values = [B]
        out = img1.copy()
        droped_out = []
        points = bc_module.Points()
        for r in residuals:
            pt1 = r[1]
            pt2 = r[2]
            point = bc_module.Point(pt1, pt2) # pt1 + r[3]
            point.make_line()
            k, b = point.get_line()
                
            pt3 = pt1 + r[3]
            out = draw.draw_line(out, k, b, color = draw.gold)
            out = draw.draw_arrow(out, pt1, pt2)
            out = draw.draw_arrow(out, pt1, pt3, color=draw.cyan)
            points.add_point(pt1, pt2)
        
        out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:02.05f}'.format(B), font_scale = fs, color = draw.blue)
        PAI, _ = points.find_PAI()
        if _:
            out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
            out = draw.draw_point(out, PAI, color=draw.purple, thickness=3)
            
        cv2.imwrite('output/{}_A.png'.format(frame_itt), out)
        out = img1.copy()
        
        y = [np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) for r in residuals]
        y.sort()
        y.reverse()
        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
        ax.set_title('|res| / |model| by points')
        ax.plot(y, color='green')
        fig.savefig('output/{}_A_g.png'.format(frame_itt))
        del fig
        for cln_itt in range(1):
            points = bc_module.Points()
            residuals.sort(key=lambda r: np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]))
            ind_max = len(residuals)
            for i in range(len(residuals)):
                r = residuals[i]
                if np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) > OUTLIER_THRESHOLD:
                    ind_max = i
                    print('drop out:', i, np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]))
                    break
#            ind_max = int(len(residuals) * 0.95)
#            for r in residuals[ind_max:]:
#                print('res = {:02.06}\tf = {:02.06}'.format(np.linalg.norm(r[4]), np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3])))
            for r in residuals[ind_max:]:
                droped_out.append(r)
            residuals = residuals[:ind_max]
            print('itt:', cln_itt, 'len:', len(residuals), 'sum =', sum_of_residuals(residuals))
            pts1 = [r[1] for r in residuals]
            pts2 = [r[2] for r in residuals]
            A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1, pts2, pai_in_frame, L) # preliminary model
            residuals = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
            B_values.append(B)
            out = img1.copy()
            out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:02.05f}'.format(B), font_scale = fs, color = draw.blue)
            for r in residuals:
                pt1 = r[1]
                pt2 = r[2]
                point = bc_module.Point(pt1, pt2)
                point.make_line()
                k, b = point.get_line()
                
                pt3 = pt1 + r[3]
                out = draw.draw_arrow(out, pt1, pt2)
                out = draw.draw_arrow(out, pt1, pt3, color=draw.gold)
#                print(k, b, flush = True)
                out = draw.draw_line(out, k, b, color = draw.gold)
                points.add_point(pt1, pt2)
            PAI, _ = points.find_PAI()
            if _:
                out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
                out = draw.draw_point(out, PAI, color=draw.purple, thickness=3)
            
            for r in droped_out:
                pt1 = r[1]
                pt2 = r[2]
                out = draw.draw_arrow(out, pt1, pt2, color=draw.red)
        
            out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:02.05f}'.format(B), font_scale = fs, color = draw.blue)
            cv2.imwrite('output/{}_B{}.png'.format(frame_itt, cln_itt), out)
            y = [np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) for r in residuals]
            y.sort()
            y.reverse()
            fig, ax = plt.subplots( nrows=1, ncols=1 ) 
            ax.set_title('|res| / |model| by points')
            ax.plot(y, color='green')
            fig.savefig('output/{}_B{}_g.png'.format(frame_itt, cln_itt))
            del fig
            s = 0
            for r in residuals:
                s += np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3])
            r_summ.append(s / len(residuals))
#            if len(r_summ) >= 2:
#                if r_summ[len(r_summ) - 2] - r_summ[len(r_summ) - 1] <= 100:
#                    pass
#        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
#        ax.plot(r_summ, color='green')
#        ax.set_title('sum (|res| / |model|) by frame')
#        fig.savefig('output/{}__all.png'.format(frame_itt))    
#        del fig
#        dif = [r_summ[i] - r_summ[i + 1] for i in range(len(r_summ) -1)]
#        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
#        ax.plot(dif, color='blue')
#        ax.set_title('diff[ sum(|res| / |model|) ] by itterations')
#        fig.savefig('output/diff{}__diffx.png'.format(frame_itt))    
#        del fig
#        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
#        ax.plot(B_values, color='red')
#        ax.set_title('B by itterations')
#        fig.savefig('output/B{}__.png'.format(frame_itt))    
#        del fig
#        T = time.time() - t0
#        print('{:02d} min {:02.03f} sec per frame'.format(int(T // 60), T % 60))
            
            
        continue
        pts1_clean = pts1
        pts2_clean = pts2
        out = img1.copy()
#        for r_i in range(len(residuals)):
#            r = residuals[r_i]
#            pt1 = r[1]     # coordinate of first  point of optical flaw
#            pt2_opt = r[2] # coordinate of second point of optical flaw
#            opt_vector = r[2] - r[1] # coordinate of vector of optical flaw
#            model_vector = r[3]      # coordinate of model vector
#            rsdl_vector = r[4]       # coordinate of residual vector
#            # criterion of interesting points
#            if np.linalg.norm(model_vector) > MIN_MODEL_VECTOR and np.linalg.norm(model_vector) < COEFF_OPT_MORETHAN_MODEL * np.linalg.norm(opt_vector) and np.linalg.norm(opt_vector - model_vector) / np.linalg.norm(model_vector) <= MAX_RESIDUAL:
#                pts1_clean.append(pt1)
#                pts2_clean.append(pt2_opt)
#                out = draw.draw_arrow(out, pt1, pt2_opt)
#                out = draw.draw_arrow(out, pt1, model_vector + pt1, color=draw.cyan)
##                x1 = pt1[0]
##                y1 = pt1[1]
#                x2 = pt1[0] + model_vector[0]
#                y2 = pt1[1] + model_vector[1]
#                k, b = sf.line_parameters(pt1, pt1 + model_vector)
#                out = draw.draw_line(out, k, b, color = draw.gold)
                
                
        points_for_pai.add_points(pts1_clean, pts2_clean)
        points_for_pai.make_lines()
        PAI, pai_bool = points_for_pai.find_PAI()#find_OF_crossing_pt(num_cycles = 30, num_rnd_lines = 20, delta = 15)
        
        if len(pts1_clean) == 0:
            print('ERROR')
            cv2.imwrite(pic_name, out)
            continue
        
        #6
        print('   steps: ', end='')
        for step_itt in range(NUMBER_OF_STEPS):
            #6.1
            print('{}'.format(step_itt), end='... ')
#            print(len(pts1), flush = True)
            M, v = build_matrix(pai[lines_itt], delta_vector, pts1_clean, pts2_clean, L)
            A, A_, B, B_, C_ = np.linalg.solve(M, v)  
            x_new = - A_ / (2 * A) # x - coordinate of new point
            y_new = - B_ / (2 * B) # y - coordinate of new point
#            print('v=', v)
            if (x_new - pai[lines_itt][0]) ** 2 + (y_new - pai[lines_itt][1]) ** 2 > step_uv ** 2: # if new point is out of range
#                print('out of range')
                x_new = pai_in_frame[0] + delta_vector[v.tolist().index(min(v))][0]
                y_new = pai_in_frame[1] + delta_vector[v.tolist().index(min(v))][1]
            print(x_new, y_new)
            out = draw.draw_arrow(out, pai[lines_itt], (x_new, y_new), draw.red, 1)
#            print('  {} {}'.format(x_new, y_new))
            pai[lines_itt] = np.array([x_new, y_new], np.float32)
            
            out = draw.draw_point(out, pai[lines_itt], 2, draw.blue, 2)
        print()
        #7
        #7.1
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1_clean, pts2_clean, pai[lines_itt], L) # preliminary model
        residuals_best = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # generation of residual array
        #7.2
        A_x_, A_y_, A_z_ = build_model_without_B(pts1_clean, pts2_clean, pai[lines_itt], L)
        #7.3
        B_var = par_change * B
        B_per = B + B_var
        residuals_pert = get_residuals(A_x_, A_y_, A_z_, B_per, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # generation of residual array
        sum_res_pert = sum_of_residuals(residuals_pert)
        sum_res_best = sum_of_residuals(residuals_best)
        eps = np.abs(B_var / np.sqrt(sum_res_pert - sum_res_best))
        file.write(' {:02d}: [{:e}, {:e}, {:e}, {:e}], {}, {}\n'.format(LINES[lines_itt], A_x, A_y, A_z, B, eps, pai[lines_itt]))
        file.flush()
        out = draw.draw_text(out, (2 * x//3, 30), 'A_x = {:02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * x//3, 60), 'A_y = {:02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * x//3, 90), 'A_z = {:02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * x//3, 120), '  B = {:02.05f}'.format(B), font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * x//3 + 240, 120), '  +/- {:02.05f}'.format(eps), font_scale = fs, color = draw.red)
        if pai_bool:
            out = draw.draw_point(out, (PAI[0], PAI[1]), radius = 5, thickness=5, color = draw.black)
            out = draw.draw_arrow(out, pai_in_frame, (PAI[0], PAI[1]), thickness=3, color = draw.black)
            out = draw.draw_point(out, (PAI[0], PAI[1]), radius = 2, thickness=3, color = draw.red)
        
        
        
        if len(data_out) < 1000 and pai_bool:
            A_x, A_y, A_z, B_new, F_x, F_y, F_z, F_f, F_o = build_model(pts1_clean, pts2_clean, PAI, L)
            data_out.append({'itt':frame_itt, 'B_old':B, 'eps_old':eps, 'speed_by_file':speed_g[frame_itt], 'B_new': B_new}) # frame_itt, B, +-, speed
            out = draw.draw_text(out, (2 * x//3, 150), 'Bnew = {:02.05f}'.format(B_new), font_scale = fs, color = draw.purple)
        print(len(data_out))
        cv2.imwrite(pic_name, out)
#        cam.read()
#        cam.read()
#        cam.read()
#        cam.read()
#        cam.read()
#        cam.read()
#        cam.read()
#        cam.read()
#        img2 = cam.read()[1]
file.close()
print('DONE')