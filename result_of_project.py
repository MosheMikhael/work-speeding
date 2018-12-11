# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:38:22 2018

@author: moshe.f
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:02:43 2018

@author: moshe.f
"""

import draw
import matplotlib.pyplot as plt
import os
import time
import sys
import cv2
import numpy as np
import collections

import point_at_infinity as pai_module
import base_classes as bc_module
import service_functions as sf
import model as mm_of_movement
import neural_network as nn


sf.del_last_files()
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
LINES = [100]
pai = np.zeros(len(LINES), object)
angle = 45
lenght_of_axis = 1_000
max_y_for_points = 400
debug = True # debug parameter
len_max = 500_000
NUMBER_OF_FRAMES = 15_000       # Number of frames that will be analysed
NUMBER_OF_STEPS = 5          # Number of steps for each frame
MIN_MODEL_VECTOR = 2         # Min norma of interesting model vector 
COEFF_OPT_MORETHAN_MODEL = 2 # Interesting points: norma of model vector > COEFF.. * norma of optical flaw vector
MAX_RESIDUAL = 1
OUTLIER_THRESHOLD = 0.5
fs = 1
dyn_points = []
for i in range(len(LINES)):
    dyn_points.append(collections.deque())
    pai[i] = pai_in_frame

filename = 'input/parsed-1750be37-5499-4c6e-9421-9bb15b277a94.txt'
K = 270
times, lon, lat, dist, speeds = sf.get_data_from_parsed_file(filename)
frame_delay = 1360
pt_min = np.array([min(lon), min(lat)])
pt_max = np.array([max(lon), max(lat)])
map_ = np.zeros((270, 270, 3)) + draw.white
pts_map = [np.array([(lon[i] - pt_min[0]) * K / (pt_max[0] - pt_min[0]), (-lat[i] + pt_max[1]) * K / (pt_max[1] - pt_min[1])]) for i in range(len(lon))]
for pt in pts_map:
    map_ = draw.draw_point(map_, (int(pt[0]), int(pt[1])), 1, thickness=1, color=draw.blue)
for i in range(len(times)):
    times[i][0] -= 13
    times[i][1] -= 55
    times[i][2] -= 28
    if times[i][0] == 1:
        times[i][0] -= 1
        times[i][1] += 60 
    m = times[i][1]
    times[i][1] -= m
    times[i][2] += m * 60
times = [t[2] for t in times]
speed_g = [-1 for i in range(frame_delay)]
lat_g = [-1 for i in range(1360)]
lon_g = [-1 for i in range(1360)]
times_g = [i / 25 for i in range(frame_delay)]
for i in range(1, len(times)):
    dt = times[i] - times[i] + 1
    s = speeds[i]
    lon_ = lon[i]
    lat_ = lat[i]
    t_ = times[i] + frame_delay / 25
    for k in range(dt * 25):
        speed_g.append(s)
        lat_g.append(lat_)
        lon_g.append(lon_)
        times_g.append(t_)
Width = img2.shape[1]#
Height = img2.shape[0]#

temp = -1
par_change = 0.01
eps = -1
best_model = -1
step_uv = 20
step_u = step_uv # 2
step_v = step_uv # 2

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
file = open(filename, 'w')
#1.1
#all_points.append(all_points_)

print('Generating arr of speed')

sa = []
Ta = []
Ba = []
#frame_delay = 5360
fr_jump = 1

times_g2 = [t / 25 for t in times_g] 
fig_GPS, ax_GPS = plt.subplots()
ax_GPS.set_title('speeding GPS')
#ax_GPS.axis([0, max(times), 0, 100])  
ax_GPS.axes.set_ylim(-3, 100)
ax_GPS.axes.set_xlim(200, 630)
ax_GPS.plot(times_g, speed_g, 'ro', color='blue', markersize=2, label='Speed')
fig_GPS.legend()
fig_GPS.savefig('output/Speed_GPS.png')
del fig_GPS, ax_GPS
with open('output/data/GPS.txt', 'w') as f:
    for i in range(len(times_g)):
        f.write('{}|{}\n'.format(times_g[i], speed_g[i]))

#sys.exit()
for frame_itt in range(NUMBER_OF_FRAMES):
    print('FRAME: {}'.format(frame_itt))
    
    file.write('frame: {}\n'.format(frame_itt))
    file.flush()
    img1 = img2
    img2 = cam.read()[1]
    
    if frame_itt < frame_delay:
        continue
#    if frame_itt % fr_jump != 0:
#        continue
    
    
    boxes_out = nn.get_features(img1)
    angles_objects = []
    if debug:
        out = img1.copy()
        out = cv2.line(out, (0, max_y_for_points + int(pai_in_frame[1])), (1920, max_y_for_points + int(pai_in_frame[1])), color=draw.cyan)
    for b in boxes_out:
        if b[4] < 8:
            angles_objects.append(np.arctan2(-pai_in_frame[1] + b[2][1], -pai_in_frame[0] + b[1][0]))
            angles_objects.append(np.arctan2(-pai_in_frame[1] + b[2][1], -pai_in_frame[0] + b[2][0]))
#        out = draw.draw_bounding_box(out, None, b[3], b[1], b[2])
    al, bt = np.pi / 4, 3 * np.pi / 4
    
    if len(angles_objects) != 0:
        al = min([np.pi / 4 if a > np.pi / 2 or a < 0 else a for a in angles_objects])
        bt = max([3 * np.pi / 4 if a > np.pi     else a for a in angles_objects])
    if al < 0 or al > np.pi / 4:
        al = np.pi / 4
    if bt < 3 * np.pi / 4:
        bt = 3 * np.pi / 4
    print('alpha =', al * 180 / np.pi, 'betta =', bt * 180 / np.pi)
    if debug:
        out = draw.draw_arrow(out, pai_in_frame, pai_in_frame + (int(3 * lenght_of_axis * np.cos(al)), int(3 * lenght_of_axis * np.sin(al))), color=draw.red,  thickness=2)
        out = draw.draw_arrow(out, pai_in_frame, pai_in_frame + (int(3 * lenght_of_axis * np.cos(bt)), int(3 * lenght_of_axis * np.sin(bt))), color=draw.blue, thickness=2)
        for an in angles_objects:
            out = draw.draw_arrow(out, pai_in_frame, pai_in_frame + (int(3 * lenght_of_axis * np.cos(an)), int(3 * lenght_of_axis * np.sin(an))), color=draw.black, thickness=1)
#    cv2.imwrite('output/_.png', out)
#    continue
    for lines_itt in range(len(LINES)):
        print(' Lines: {:02d}'.format(LINES[lines_itt]), end=' -> ')
        pic_name = 'output/{:02d}_lines={:03d}.png'.format(frame_itt, LINES[lines_itt])
#        out = img1.copy()
#        out = draw.draw_point(out, pai[lines_itt], 5, draw.red, 5)
#        out = cv2.ellipse(out, (pai_in_frame[0], pai_in_frame[1]), (lenght_of_axis, lenght_of_axis), 0, angle, 180 - angle, color=draw.cyan)
#        out = cv2.line(out, (_x1_, _y_), (_x2_, _y_), color=draw.cyan)
#        _y_ = int(pai_in_frame[1]) + max_y_for_points
#        _x1_ = int(x // 4)
#        _x2_ = int(3 * x // 4)
        
        #1.2
        all_points = []
        for IND_LINES in range(len(LINES)):
            lines = LINES[IND_LINES]
            all_points_, count_of_interesting_points = sf.get_points(lenght_of_axis, np.pi / 2 - al, bt - np.pi / 2 ,lines, max_y_for_points, pai_in_frame) # generating vector of all points
#            for pts_ in all_points_:
#                for pt in pts_:
#                    out = draw.draw_point(out, pt, 1)
#        cv2.imwrite('output/_.png', out)
        all_points.append(all_points_)
        interesting_points = sf.intensity_threshould_filter_of_points(img1, all_points[lines_itt], intensity_threshould) # generation an interesting points for current frame
        tmp_map = map_.copy()
        lat_current = lat_g[frame_itt]
        lon_current = lon_g[frame_itt]
        
        pt_current_coordinate = np.array([(lon_current - pt_min[0]) * K / (pt_max[0] - pt_min[0]), (-lat_current + pt_max[1]) * K / (pt_max[1] - pt_min[1])])
        tmp_map = draw.draw_point(tmp_map, pt_current_coordinate, 3, thickness=2, color=draw.red)
        if len(interesting_points) == 0:
            print('ERROR')
            if debug:
                out[50: 320,1630:1900] = tmp_map
                cv2.imwrite(pic_name, out)
            continue
        #2
        mod_pts = sf.modification_points(interesting_points)
        pts1, pts2 = pai_module.find_opt_flow_lk_with_points(img1, img2, mod_pts)
        if len(pts1) == 0:
            print('ERROR')
            if debug:
                out[50: 320,1630:1900] = tmp_map
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
        
#            print(classes[class_ids[i_]], 'was drawed border')
        t0 = time.time()
        pts1_tem = []
        pts2_tem = []
        for i in range(len(pts1)):
            pt1 = pts1[i]
            pt2 = pts2[i]
            o = True
            for box in boxes_out:
                box_pt1 = box[1]
                box_pt2 = box[2]
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
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1, pts2, pai_in_frame, L) # preliminary model
        #4
        residuals = mm_of_movement.get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
#        if debug:
#            out = img1.copy()
        droped_out = []
        points = bc_module.Points()
        if debug:
            for box in boxes_out:
                cl_name = box[0]
                pt1 = box[1]
                pt2 = box[2]
                cID = box[4]
                if cID < 8:
                    out = draw.draw_bounding_box(out, cl_name, draw.red, pt1, pt2, 3)
                else:
                    out = draw.draw_bounding_box(out, cl_name, draw.blue, pt1, pt2, 3)
            
        for r in residuals:
            pt1 = r[1]
            pt2 = r[2]
            point = bc_module.Point(pt1, pt2) # pt1 + r[3]
            point.make_line()
            k, b = point.get_line()
                
            pt3 = pt1 + r[3]
            if debug:
#                out = draw.draw_line(out, k, b, color = draw.gold)
                out = draw.draw_arrow(out, pt1, pt2)
                out = draw.draw_arrow(out, pt1, pt3, color=draw.cyan)
            points.add_point(pt1, pt2)
#        if debug:
#            out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
#            out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
#            out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
#            out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:02.05f}'.format(B), font_scale = fs, color = draw.blue)
        PAI, _ = points.find_PAI()
        if _:
            if debug:
                out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
                out = draw.draw_point(out, PAI, color=draw.purple, thickness=3)
        if debug:
            out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:+02.05f}'.format(A_x), font_scale = fs, color = draw.white, line_type=6)
            out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:+02.05f}'.format(A_y), font_scale = fs, color = draw.white, line_type=6)
            out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:+02.05f}'.format(A_z), font_scale = fs, color = draw.white, line_type=6)
            out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:+02.05f}'.format(B), font_scale = fs, color = draw.white, line_type=6)
            out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:+02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:+02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:+02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:+02.05f}'.format(B), font_scale = fs, color = draw.blue)
            out = draw.draw_text(out, (2 * Width//3, Height - 100), 'Speed = {:02.04f} km/h'.format(speed_g[frame_itt]), font_scale = fs, color = draw.white, line_type=6)
            out = draw.draw_text(out, (2 * Width//3, Height - 100), 'Speed = {:02.04f} km/h'.format(speed_g[frame_itt]), font_scale = fs, color = draw.dark_green)
            out = draw.draw_arrow(out, pai_in_frame, pai_in_frame + (int(3 * lenght_of_axis * np.cos(al)), int(3 * lenght_of_axis * np.sin(al))), color=draw.black, thickness=2)
            out = draw.draw_arrow(out, pai_in_frame, pai_in_frame + (int(3 * lenght_of_axis * np.cos(bt)), int(3 * lenght_of_axis * np.sin(bt))), color=draw.white, thickness=2)
            out[50: 320,1630:1900] = tmp_map
            cv2.imwrite('output/{}_A.png'.format(frame_itt), out)
            out = img1.copy()
            out = cv2.line(out, (0, max_y_for_points + int(pai_in_frame[1])), (1920, max_y_for_points + int(pai_in_frame[1])), color=draw.cyan)
        if debug:
            for box in boxes_out:
                cl_name = box[0]
                pt1 = box[1]
                pt2 = box[2]
                if box[4] < 8:
                    out = draw.draw_bounding_box(out, None, draw.red, pt1, pt2, 3)
#        if debug:            
#            y = [np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) for r in residuals]
#            y.sort()
#            y.reverse()
#            fig, ax = plt.subplots( nrows=1, ncols=1 ) 
#            ax.set_title('|res| / |model| by points')
#            ax.plot(y, color='green')
#            fig.savefig('output/{}_A_g.png'.format(frame_itt))
#            del fig
        sa.append(speed_g[frame_itt])
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
            print('itt:', cln_itt, 'len:', len(residuals), 'sum =', mm_of_movement.sum_of_residuals(residuals))
            pts1 = [r[1] for r in residuals]
            pts2 = [r[2] for r in residuals]
            if len(pts1) == 0:
                print('All lines is outliers!!!')
                continue
#                Ba.append(10)
            A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1, pts2, pai_in_frame, L) # preliminary model
            
            residuals = mm_of_movement.get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
#            Ba[len(Ba) - 1] = -30 * B
            res_avr = mm_of_movement.sum_of_residuals(residuals)
            n_pts = len(residuals)
            B_v = -27 * B
            T_v = frame_itt / 25
            Ba.append(B_v)
            Ta.append(T_v)
                       
            if debug:
                out = img1.copy()
                out = cv2.line(out, (0, max_y_for_points + int(pai_in_frame[1])), (1920, max_y_for_points + int(pai_in_frame[1])), color=draw.cyan)
#                for box in boxes_out:
#                    cl_name = box[0]
#                    pt1 = box[1]
#                    pt2 = box[2]
#                    out = draw.draw_bounding_box(out, cl_name, draw.red, pt1, pt2, 3)
            for r in residuals:
                pt1 = r[1]
                pt2 = r[2]
                point = bc_module.Point(pt1, pt2)
                point.make_line()
                k, b = point.get_line()
                
                pt3 = pt1 + r[3]
                if debug:
                    out = draw.draw_arrow(out, pt1, pt2)
                    out = draw.draw_arrow(out, pt1, pt3, color=draw.cyan)
                    out = draw.draw_line(out, k, b, color = draw.gold)
                points.add_point(pt1, pt2)
            PAI, _ = points.find_PAI()
            if _ and debug:
                out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
                out = draw.draw_point(out, PAI, color=draw.purple, thickness=3)
            if debug:
                for r in droped_out:
                    pt1 = r[1]
                    pt2 = r[2]
                    out = draw.draw_arrow(out, pt1, pt2, color=draw.red)
            B_median7 = []
            leng = 7
            for i in range(len(Ba)):
                if i < leng:
                    B_median7.append(Ba[i])
                else:
                    B_median7.append(np.median(Ba[i-leng:i]))
            B = B_median7[len(B_median7) - 1]
            if debug:
                out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:+02.05f}'.format(A_x), font_scale = fs, color = draw.white, line_type=6)
                out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:+02.05f}'.format(A_y), font_scale = fs, color = draw.white, line_type=6)
                out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:+02.05f}'.format(A_z), font_scale = fs, color = draw.white, line_type=6)
                out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:+02.05f}'.format(B*-27), font_scale = fs, color = draw.white, line_type=6)
                out = draw.draw_text(out, (2 * Width//3, 30), 'A_x = {:+02.05f}'.format(A_x), font_scale = fs, color = draw.blue)
                out = draw.draw_text(out, (2 * Width//3, 60), 'A_y = {:+02.05f}'.format(A_y), font_scale = fs, color = draw.blue)
                out = draw.draw_text(out, (2 * Width//3, 90), 'A_z = {:+02.05f}'.format(A_z), font_scale = fs, color = draw.blue)
                out = draw.draw_text(out, (2 * Width//3, 120), '  B = {:+02.05f}'.format(B*-27), font_scale = fs, color = draw.blue)
                out = draw.draw_text(out, (2 * Width//3, Height - 100), 'Speed by GPS = {:02.04f} km/h'.format(speed_g[frame_itt]), font_scale = fs, color = draw.white, line_type=6)
                out = draw.draw_text(out, (2 * Width//3, Height - 100), 'Speed by GPS = {:02.04f} km/h'.format(speed_g[frame_itt]), font_scale = fs, color = draw.dark_green)
                out[50: 320,1630:1900] = tmp_map
                cv2.imwrite('output/{}_B{}.png'.format(frame_itt, cln_itt), out)
#                y = [np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) for r in residuals]
#                y.sort()
#                y.reverse()
                
                
                if debug:
                    fig_B, ax_B = plt.subplots()
                    ax_B.set_title('-30 * B at median 7')
                    ax_B.axis([200, 630, -5, 100])        
                    ax_B.plot(Ta, B_median7, 'ro', color='blue', markersize=2)
                    fig_B.savefig('output/{}_B{}_g_med7.png'.format(frame_itt, cln_itt))
                    del fig_B, ax_B
                
                
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('-30 * B')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(Ta, Ba, 'ro', color='green', markersize=2)
#                fig_B.savefig('output/{}_B{}_g_all.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('speeding B: N > 10')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(T10, B10, 'ro', color='green', markersize=3)
#                fig_B.savefig('output/{}_B{}_g_N10.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('speeding B: N > 20')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(T20, B20, 'ro', color='green', markersize=3)
#                fig_B.savefig('output/{}_B{}_g_N20.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#                
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('speeding B: avr{res} > 0.1')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(Tavr01, Bavr01, 'ro', color='green', markersize=3)
#                fig_B.savefig('output/{}_B{}_g_avr01.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('speeding B: avr{res} > 0.2')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(Tavr02, Bavr02, 'ro', color='green', markersize=3)
#                fig_B.savefig('output/{}_B{}_g_avr02.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#                fig_B, ax_B = plt.subplots()
#                ax_B.set_title('speeding B: avr{res} > 0.3')
#                ax_B.axis([200, 630, -5, 100])                
#                ax_B.plot(Tavr03, Bavr03, 'ro', color='green', markersize=3)
#                fig_B.savefig('output/{}_B{}_g_avr03.png'.format(frame_itt, cln_itt))
#                del fig_B, ax_B
#            s = 0
#            for r in residuals:
#                s += np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3])
#            r_summ.append(s / len(residuals))
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
        with open('output/data/B.txt', 'w') as f:
            for i in range(len(B_median7)):
                f.write('{}|{}\n'.format(Ta[i], B_median7[i]))
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
            M, v = mm_of_movement.build_matrix(pai[lines_itt], delta_vector, pts1_clean, pts2_clean, L)
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
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1_clean, pts2_clean, pai[lines_itt], L) # preliminary model
        residuals_best = mm_of_movement.get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # generation of residual array
        #7.2
        A_x_, A_y_, A_z_ = mm_of_movement.build_model_without_B(pts1_clean, pts2_clean, pai[lines_itt], L)
        #7.3
        B_var = par_change * B
        B_per = B + B_var
        residuals_pert = mm_of_movement.get_residuals(A_x_, A_y_, A_z_, B_per, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # generation of residual array
        sum_res_pert = mm_of_movement.sum_of_residuals(residuals_pert)
        sum_res_best = mm_of_movement.sum_of_residuals(residuals_best)
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
            A_x, A_y, A_z, B_new, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1_clean, pts2_clean, PAI, L)
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