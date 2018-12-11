# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:02:43 2018

@author: moshe.f
"""

import matplotlib.pyplot as plt
import time
import cv2
import numpy as np

import sys
import os
import collections

import draw
import point_at_infinity as pai_module
import base_classes as bc_module
import service_functions as sf
import model as mm_of_movement
import neural_network as nn

def time_tressing(t0, string):
    T = time.time() - t0
    print(string + ': ' + str(T) + ' sec')
    return T
    
def log(nfr, time_start_itt, tGPS, loc, gpsSpeed, A_x=None, A_y=None, A_z=None, B=None, ofSpeed=None, medianSpeed=None, pai=None, err=None):
    t0 = time.time()
    pname = 'output/log.txt'
    line1 = 'frame #{:05d}:\n'.format(nfr)
    line2 = None
    pai_ = None
    if pai is not None:
        pai_ = (pai[0], pai[1])
    if err is not None:
        line3 = '\tErrors: {}\n'.format(err)
    line2 = '\tcurrent time: {}\n\t\ttime per frame: {:f}\n\t\ttime by GPS: {}\n\t\tloc={}\n\t\t(Ax,Ay,Az,B)=({},{},{},{})\n\t\tsALG={}\n\t\tsAM7={}\n\t\tsGPS={}\n\t\tpai={}\n'.format(time.ctime(), time.time() - time_start_itt, tGPS, loc, A_x, A_y, A_z, B, ofSpeed, medianSpeed, gpsSpeed, pai_)
    with open(pname, 'a') as f:
        f.write(line1)
        f.write(line2)
        if err is not None:
            f.write(line3)
    tt_log.append(time_tressing(t0, '\t\tlogging'))
        
def make_result(img, nfr, time_start_itt, Map, tGPS, loc=None, gpsSpeed=None, residuals=None, A_x=None, A_y=None, A_z=None, B=None,
                ofSpeed=None, medianSpeed=None, pai=None, err=None, time_per_frame=[], debug=True, debug_graph=True):
    log(nfr, time_start_itt, tGPS, loc, gpsSpeed, A_x, A_y, A_z, B, ofSpeed, medianSpeed, pai, err)
    if debug_graph:
        t0 = time.time()
        while len(time_per_frame) + 1 != len(Ta):
            time_per_frame.append(None)
        time_per_frame.append(time.time() - time_start_itt)
        f, a = plt.subplots()
        a.set_title('Time per frame')
        a.plot(Ta, time_per_frame, 'ro', color='red', markersize=3, label='time')
        f.legend()
        f.savefig('output/Time_per_frame.png')
        tt_draw_graph.append(time_tressing(t0, '\t\tdrawing graph'))
#    f.clf()
    if debug:
        t0 = time.time()
        lt = 6
        out = img.copy()
        Width  = img.shape[1]#
        Height = img.shape[0]#
        fs = Height / Width
        pic_name = 'output/{:05d}.png'.format(nfr)
        if residuals is not None:
            for r in residuals:
                pt1 = r[1]
                pt2 = r[2]
                pt3 = pt1 + r[3]
                out = draw.draw_arrow(out, pt1, pt2)
                out = draw.draw_arrow(out, pt1, pt3, color=draw.cyan)
        tt_draw_of.append(time_tressing(t0, '\t\tof drawing'))
        t0 = time.time()
        a = Map.shape[1]
        dw = int(0.05 * Width)
        dh = int(0.05 * Height)
        out = cv2.circle(out, (int(Width - dw - a / 2 - 3), int(dh + a / 2)) , 3 + int(np.sqrt(2) * a / 2), COLOR_OF_MAP_BACKGRAUND, -1)
        out[dh:dh + a, Width - dw - a:Width - dw] = Map
        line1, line2, line3 = '???', '???', '???'
        if ofSpeed is not None:
            line1 = '{:+02.03f}'.format(ofSpeed)
        if medianSpeed is not None:
            line2 = '{:+02.03f}'.format(medianSpeed)
        if gpsSpeed is not None:
            line3 = '{:+02.03f}'.format(gpsSpeed)
    
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 200), 'Speed by ALG = ' + line1 + ' km/h', font_scale = fs, color = draw.white, line_type=lt)
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 200), 'Speed by ALG = ' + line1 + ' km/h', font_scale = fs, color = draw.blue)
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 150), 'Speed by AM7 = ' + line2 + ' km/h', font_scale = fs, color = draw.white, line_type=lt)
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 150), 'Speed by AM7 = ' + line2 + ' km/h', font_scale = fs, color = draw.dark_red)
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 100), 'Speed by GPS = ' + line3 + ' km/h', font_scale = fs, color = draw.white, line_type=lt)
        out = draw.draw_text(out, (2 * Width // 3, Height - fs * 100), 'Speed by GPS = ' + line3 + ' km/h', font_scale = fs, color = draw.dark_green)
    
        cv2.imwrite(pic_name, out)
        tt_draw_interface.append(time_tressing(t0, '\t\tanother drawing'))

def get_median(value, size=7, max_memory=10, arr=[]):
    if len(arr) == max_memory:
        arr.remove(arr[0])
    arr.append(value)
    actual = []
    for d in arr:
        if d is not None:
            actual.append(d)
    if len(actual) < size:
        return None
    return np.median(actual)

#deleting old results        
sf.del_last_files()
with open('output/log.txt', 'w') as file:
    file.write('PROGRAM WAS STARTED AT {}\n'.format(time.ctime()))
video_adr = 'input/1750be37-5499-4c6e-9421-9bb15b277a94.mp4'
#video_adr = 'out.avi'
filename = 'input/parsed-1750be37-5499-4c6e-9421-9bb15b277a94.txt'
debug = True 
debug_graph = True
cam = cv2.VideoCapture(video_adr)
_, img2 = cam.read()  

y, x, z = img2.shape
pai_in_frame = np.array([x // 2 - 50,  y * 0.4], np.int)
Width = img2.shape[1]#
Height = img2.shape[0]#
fs = 1
COLOR_OF_MAP_BACKGRAUND = draw.white
frame_delay = 1360 # frames, that was not by GPS

# MAGIC NUMBERS
B_factor = -27 * Width / 1920
intensity_threshould = 25 # used in generating interesting points
L = 10_000 # focus lenght
LINES = 100
angle = 70
lenght_of_axis = 1_000
max_y_for_points = 0.66 * Height / 2 
len_max = 500_000
NUMBER_OF_FRAMES = 14_500       # Number of frames that will be analysed
NUMBER_OF_STEPS = 5          # Number of steps for each frame
MIN_MODEL_VECTOR = 2         # Min norma of interesting model vector 
COEFF_OPT_MORETHAN_MODEL = 2 # Interesting points: norma of model vector > COEFF.. * norma of optical flaw vector
MAX_RESIDUAL = 1
OUTLIER_THRESHOLD = 0.5
delta_fr = 480
# Data from file
times, lon, lat, dist, speeds = sf.get_data_from_parsed_file(filename)
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
times   = [t[2] for t in times]
speed_g = [-1 for i in range(frame_delay)]
lat_g   = [-1 for i in range(frame_delay)]
lon_g   = [-1 for i in range(frame_delay)]
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
# Map Generating
pt_min = np.array([min(lon), min(lat)])
pt_max = np.array([max(lon), max(lat)])
K = int(Height / 4) # window size for map
map_ = np.zeros((K, K, 3)) + COLOR_OF_MAP_BACKGRAUND
pts_map = [np.array([(lon[i] - pt_min[0]) * K / (pt_max[0] - pt_min[0]), (-lat[i] + pt_max[1]) * K / (pt_max[1] - pt_min[1])]) for i in range(len(lon))]
for pt in pts_map:
    map_ = draw.draw_point(map_, (int(pt[0]), int(pt[1])), 1, thickness=1, color=draw.blue)

sa = []
Ta = []
Ba = []
B_median7 = []
out = -1
tmp_map = -1
PAI = -1
#time test
tt_reading = []
tt_nn = []
tt_draw_rec = []
tt_bb_filtering = []
tt_draw_of = []
tt_log = []
tt_draw_graph = []
tt_draw_another = []
tt_draw_interface = []
tt_points_gen = []
tt_of_gen = []
tt_model1 = []
tt_model2 = []
tt_of_filtering = []
tt_pai = []
vehicles_classes = ['car', 'motorcycle', 'bus', 'truck', 'train']

for frame_itt in range(1, NUMBER_OF_FRAMES):
    t0 = time.time()
    plt.close('all')
    print('FRAME: {}'.format(frame_itt))
    pic_name = 'output/{:05d}.png'.format(frame_itt)
    img1 = img2
    img2 = cam.read()[1]
    tt_reading.append(time_tressing(t0, '\t\treading'))
    
    # for skiping video (by defoult = 1360)
    if frame_itt < 1360         : 
        print('\tpass')
        continue
    T_v = frame_itt / 25 # - 120
    Ta.append(T_v)
    sa.append(speed_g[frame_itt + delta_fr])
    angles_objects = [] 
    boxes_main = []
    time_start_itt = time.time()
    boxes_out = nn.get_features(img1)
    tt_nn.append(time_tressing(time_start_itt, '\t\tneural network'))
    t0 = time.time()
    if debug:
        out = img1.copy()
        for box in boxes_out:
            cl_name = box[0]
            pt1 = box[1]
            pt2 = box[2]
            if cl_name in vehicles_classes:
                out = draw.draw_bounding_box(out, cl_name, draw.red, pt1, pt2, 3)
            else:
                out = draw.draw_bounding_box(out, cl_name, draw.blue, pt1, pt2, 3)
    tt_draw_rec.append(time_tressing(t0, '\t\tbounding box drawing'))
    t0 = time.time()
    for box in boxes_out:
        cl_name = box[0]
        if cl_name in vehicles_classes:
            angles_objects.append(np.arctan2(-pai_in_frame[1] + box[2][1], -pai_in_frame[0] + box[1][0]))
            angles_objects.append(np.arctan2(-pai_in_frame[1] + box[2][1], -pai_in_frame[0] + box[2][0]))
            boxes_main.append(box)
    boxes_out = boxes_main # looking only on actual objects
    tt_bb_filtering.append(time_tressing(t0, '\t\tbb filtering'))
    if frame_itt < frame_delay:
        make_result(out, frame_itt, time_start_itt, map_, times_g[frame_itt], err='Cutted Frame', debug=debug, debug_graph=debug_graph)
        continue

    # angles searching
    t0 = time.time()
    al, bt = np.pi / 4, 3 * np.pi / 4
    if len(angles_objects) != 0:
        al = min([np.pi / 4 if a > np.pi / 2 or a < 0 else a for a in angles_objects])
        bt = max([3 * np.pi / 4 if a > np.pi     else a for a in angles_objects])
    if al < 0 or al > np.pi / 4:
        al = np.pi / 4
    if bt < 3 * np.pi / 4:
        bt = 3 * np.pi / 4
    if debug:
        t1 = time.time()
        pt_l0 = np.array([0, max_y_for_points]) + pai_in_frame
        pt_l1 = np.array([Width, max_y_for_points]) + pai_in_frame
        pt_l0 = (int(0), int(pt_l0[1]))
        pt_l1 = (int(Width), int(pt_l1[1]))
        R = np.sqrt(Height ** 2 + Width ** 2)
        ptA = (int(R * np.cos(al)) + pai_in_frame[0], int(R * np.sin(al)) + pai_in_frame[1])
        ptB = (int(R * np.cos(bt)) + pai_in_frame[0], int(R * np.sin(bt)) + pai_in_frame[1])
        ptO = (int(pai_in_frame[0]), int(pai_in_frame[1]))
        out = cv2.line(out, pt_l0, pt_l1, color=draw.cyan)
        out = cv2.line(out, ptO, ptA, color=draw.cyan)
        out = cv2.line(out, ptO, ptB, color=draw.cyan)
        tt_draw_another.append(time_tressing(t1, '\t\tdrawing axis'))
    all_points, count_of_interesting_points = sf.get_points(lenght_of_axis, np.pi / 2 - al, bt - np.pi / 2 ,LINES, max_y_for_points, pai_in_frame) # generating vector of all points
    interesting_points = sf.intensity_threshould_filter_of_points(img1, all_points, intensity_threshould) # generation an interesting points for current frame
    tt_points_gen.append(time_tressing(t0, '\t\tpoints gen'))
#drawing map
    if debug:
        tmp_map = map_.copy()
        lat_current = lat_g[frame_itt + delta_fr]
        lon_current = lon_g[frame_itt + delta_fr]
        pt_current_coordinate = np.array([(lon_current - pt_min[0]) * K / (pt_max[0] - pt_min[0]), (-lat_current + pt_max[1]) * K / (pt_max[1] - pt_min[1])])
        tmp_map = draw.draw_point(tmp_map, pt_current_coordinate, 3, thickness=2, color=draw.red)
    
    #add B = None
    if len(interesting_points) == 0:
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    ofSpeed=None, medianSpeed=get_median(None), err='There are not interesting points', debug=debug, debug_graph=debug_graph)
        continue
#OPTICAL FLOW
    t0 = time.time()
    mod_pts = sf.modification_points(interesting_points)
    try:
        pts1, pts2 = pai_module.find_opt_flow_lk_with_points(img1, img2, mod_pts)
    except:
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    ofSpeed=None, medianSpeed=get_median(None), err='Faild in optical flow', debug=debug, debug_graph=debug_graph)
        continue
    if len(pts1) == 0:
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    ofSpeed=None, medianSpeed=get_median(None), err='No optical flow', debug=debug, debug_graph=debug_graph)
        continue
    #5 CLEANING (dropout points that in objects)
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
        
#    print('\t', len(pts1) - len(pts1_tem), 'was filtered by objects.')
    pts1 = pts1_tem
    pts2 = pts2_tem
    tt_of_gen.append(time_tressing(t0, '\t\tof gen'))
    #3
    if len(pts1) == 0:
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    ofSpeed=None, medianSpeed=get_median(None), err='Bad optical flow', debug=debug, debug_graph=debug_graph)
        continue
    try:
        t0 = time.time()
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1, pts2, pai_in_frame, L) # preliminary model
        residuals = mm_of_movement.get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
        tt_model1.append(time_tressing(t0, '\t\tmodel1'))
    except:
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    ofSpeed=None, medianSpeed=get_median(None), err='Mistake in matrix #1', debug=debug, debug_graph=debug_graph)
        continue
    
#    t0 = time.time()
#    pai_points = bc_module.Points()
#    pai_points.add_points(pts1, pts2)
#    PAI, check = pai_points.find_PAI()
#    tt_pai.append(time_tressing(t0, '\t\tpai finding'))
#    if not check:
#        if debug and check:
#            out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
#            out = draw.draw_point(out, PAI, color=draw.purple, thickness=4)
    t0 = time.time()
    residuals.sort(key=lambda r: np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]))
    ind_max = len(residuals)
    for i in range(len(residuals)):
        r = residuals[i]
        if np.linalg.norm(r[2] - r[1] - r[3]) / np.linalg.norm(r[3]) > OUTLIER_THRESHOLD:
            ind_max = i
            break
    if debug:
        rr = residuals[ind_max:]
#    out = img2.copy()
        for r in rr:
            pt1 = r[1]
            pt2 = r[2]
            pt3 = pt1 + r[3]
            out = draw.draw_arrow(out, pt1, pt2, color=draw.red)
            out = draw.draw_arrow(out, pt1, pt3, color=draw.gold)
#    cv2.imwrite('output/red{}.png'.format(frame_itt), out)
    residuals = residuals[:ind_max]
    pts1 = [r[1] for r in residuals]
    pts2 = [r[2] for r in residuals]
    tt_of_filtering.append(time_tressing(t0, '\t\tof filtering'))
    if len(pts1) == 0:
#        print('All lines is outliers!!!')
        B_v = B_factor * B
        T_v = frame_itt / 25 # - 120
        Ba.append(B_v)
        Ta.append(T_v)
        sa.append(speed_g[frame_itt])
        Bm7_v = get_median(B_v)
        B_median7.append(Bm7_v)
        make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                    None, A_x, A_y, A_z, B, ofSpeed=B_v, medianSpeed=Bm7_v, err='All optical flow is outlier', debug=debug, debug_graph=debug_graph)
        continue
    try:
        t0 = time.time()
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = mm_of_movement.build_model(pts1, pts2, pai_in_frame, L) # preliminary model
        residuals = mm_of_movement.get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1, pts2) # generation of residual array
        tt_model2.append(time_tressing(t0, '\t\tmodel2'))
    except:
        log(frame_itt, time_start_itt, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], A_x, A_y, A_z, B, -27 * B, B_median7[len(B_median7) - 1], err='Mistake in matrix #2')
    
    t0 = time.time()
    pai_points = bc_module.Points()
    pai_points.add_points(pts1, pts2)
    PAI, check = pai_points.find_PAI()
    tt_pai.append(time_tressing(t0, '\t\tpai finding'))
    if debug and check:
        t0 = time.time()
        out = draw.draw_arrow(out, pai_in_frame, PAI, color=draw.purple)
        out = draw.draw_point(out, PAI, color=draw.purple, thickness=4)
        tt_draw_another[len(tt_draw_another) - 1] += time_tressing(t0, '\t\tdrawing pai')
#    res_avr = mm_of_movement.sum_of_residuals(residuals)
#    n_pts = len(residuals)
    B_v = B_factor * B
    while len(Ba) + 1 != len(Ta):
        Ba.append(None)
    Ba.append(B_v)
    Bm7_v = get_median(B_v)
    while len(B_median7) + 1 != len(Ta):
        B_median7.append(None)
    B_median7.append(Bm7_v)
    make_result(out, frame_itt, time_start_itt, tmp_map, times_g[frame_itt + delta_fr], (lon_g[frame_itt + delta_fr], lat_g[frame_itt + delta_fr]), speed_g[frame_itt + delta_fr], 
                residuals, A_x, A_y, A_z, B, ofSpeed=B_v, medianSpeed=Bm7_v, pai=PAI if check else None, debug=debug, debug_graph=debug_graph, err=None if check else 'Point of infinity isn`t find')
    if debug_graph:
        t0 = time.time()
        fig_B, ax_B = plt.subplots()
        ax_B.set_title('B')
        ax_B.set_ylim(bottom=-10, top=120)
        ax_B.plot(Ta, sa, 'ro', color='green', markersize=3)
        ax_B.plot(Ta, Ba, 'ro', color='blue', markersize=2)
        ax_B.plot(Ta, B_median7, 'ro', color='red', markersize=1)
        fig_B.legend(('Speed GPS', 'factor * B', 'Median 7'))
        fig_B.savefig('output/B.png')
        tt_draw_graph.append(time_tressing(t0, '\t\tdrawing graph of speed'))

print('ALGORITHM:')
print('\treading:', np.mean(tt_reading))
print('\tneural:',np.mean(tt_nn))
print('\tanalise bb:',np.mean(tt_bb_filtering))
print('\tpoints gen:',np.mean(tt_points_gen))
print('\tOF:',np.mean(tt_of_gen))
print('\tOF filt:',np.mean(tt_of_filtering))
print('\tmodel1:',np.mean(tt_model1))
print('\tmodel2:',np.mean(tt_model2))
print('\tpai finding1:',np.mean(tt_pai))
print('\tloging:',np.mean(tt_log))
print('summary time:', sum([np.mean(tt_reading),      np.mean(tt_nn), 
                            np.mean(tt_bb_filtering), np.mean(tt_points_gen), 
                            np.mean(tt_of_gen),       np.mean(tt_of_filtering), 
                            np.mean(tt_model1),       np.mean(tt_model2), 
                            np.mean(tt_pai),          np.mean(tt_log)]))

print('DRAWING:')
print('\tdrawing rec:',           np.mean(tt_draw_rec))
print('\tdrawing OF:',            np.mean(tt_draw_of))
print('\tdrawing graph:',         np.mean(tt_draw_graph))
print('\tdrawing data on frame:', np.mean(tt_draw_interface))
print('\tdrawing another:',       np.mean(tt_draw_another))
print('DONE')
