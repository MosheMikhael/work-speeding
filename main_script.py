# -*- coding: utf-8 -*-
import service_functions as sf
import cv2
import sys
import collections
import numpy as np
import draw
import matplotlib.pyplot as plt
import time
import point_at_infinity as pai
import base_classes as bc

def generate_straight_lines_array(frames, m, K, v_speedomether=None, v_limit=10, 
                                  alpha=0.8, mask=None):
    print('\t generatin straight lines array:')
    file = open('points_data.txt', 'w')
    r_sq = sf.get_mahalanobis_distance_sq_by_probability(alpha)
    r = np.sqrt(r_sq)
    K_inv = np.linalg.inv(K)
    frame_nums = len(frames)
    status = np.zeros((frame_nums), dtype=np.int)
    times = collections.deque(maxlen=frame_nums)
    for i in range(1, frame_nums):
        t0 = time.time()
        area1 = sf.get_box(frames[i - 1], mask)
        area2 = sf.get_box(frames[i], mask)
        if v_speedomether is not None:
            if v_speedomether[i] < v_limit:
                continue
        pt, st = pai.find_OF_crossing_pt(area1, area2, method='lk')
        if not st:
            status[i] = status[i - 1]
            continue
        pt += mask[0]
        r_point_sq = sf.get_mahalanobis_distance_sq(pt, m, K_inv)
        r_point = np.sqrt(r_point_sq)
        #print('{:.2f} vs {:.2f}'.format(r_point, 3 * r))
        if r_point < 3 * r:
            status[i] = 1
        file.write('{}|{:.02f} vs {:.02f} ==> {}\n'.format(pt, r_point, 3 * r, status[i]))
        tk = time.time()
        T = tk - t0
        times.append(T)
        T = (sum(times) / len(times)) * (frame_nums - i)
        t_min = T // 60
        t_sec = T % 60
        print('\tTime left: {}min {:.02f}sec'.format(int(t_min), t_sec))
    file.close()    
    return status
        
if __name__ == '__main__':
    data_c = 'calibration_video.mp4'  # calibration video
    data = 'test1.mp4'                # video
    min_analysed_frame_nb = 300
    alpha = 0.8
    
    video = cv2.VideoCapture(data)
    lengh = np.int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds = collections.deque(maxlen=lengh)
    with open('speeds.txt') as file:
        for line in file:
            speeds.append(np.float(line.split('|')[1]))
    speeds = np.array(speeds)
    
    f = video.read()[1]
    img_dim = f.shape 
    r_sq = sf.get_mahalanobis_distance_sq_by_probability(alpha)
    r = np.sqrt(r_sq)
    cap = bc.Capture(data_c)
    #m, K, mask = pai.get_pt_at_infinity(cap, img_dim, min_analysed_frame_nb, method = 'lk', mask_width_ratio = 1/2, trace = False, save_points = False)
    #sf.saveCalibrationOutput(m, K, mask, 'calibration.txt')
   
    m, K, mask = sf.readCalibrationOutput('calibration.txt')
    #sys.exit()   
    K_inv = np.linalg.inv(K)    
    frames = collections.deque(maxlen=lengh)
    for i in range(1, 1500 + 1):
        print('  {} / {}'.format(i, lengh+1), flush=True)
        frames.append(f)
        _, f = video.read()
        
#    status = generate_straight_lines_array(frames, m, K, speeds, mask=mask)
#    
#    r0 = status
#    file = open('r0.txt', 'w')
#    file.write(status.__str__())
#    file.close()
#    status = sf.mean_filter(status)
#    file = open('r1.txt', 'w')
#    r1 = status
#    file.write(status.__str__())
#    file.close()
##    status = np.rint(status)
#    status = np.array(np.rint(status), dtype=np.int)
#    r2 = status
#    file = open('r2.txt', 'w')
#    file.write(status.__str__())
#    file.close()
 
    segments = []
    swtch_seg = False
    start, end = 0, 0
    d = 10
    v_mtr = collections.deque(maxlen=lengh)
    v_gps = collections.deque(maxlen=lengh)
    for s in speeds:
        v_gps.append(s + 10*np.random.random() - 5)
        v_mtr.append(s + d)
    v_gps = np.array(v_gps)
    v_mtr = np.array(v_mtr)    
    
    status = generate_straight_lines_array(frames, m, K, v_mtr, mask=mask)
    status = sf.mean_filter(status)
    status = np.array(np.rint(status), dtype=np.int)
    
    for i in range(len(status)):  # segments
        if status[i] == 1:
            if not swtch_seg:
                swtch_seg = True
                start = i
        else:
            if swtch_seg:
                swtch_seg = False
                segments.append([start, i])
                
    file = open('r3.txt', 'w')
    file.write(segments.__str__())
    file.close()
    
    mV_gps = collections.deque(maxlen=len(segments))
    for seg in segments:
        mV_gps.append(np.mean(v_gps[seg[0]:seg[1]]))
    
    deltas = collections.deque(maxlen=len(segments))
    for i in range(len(segments)):
        deltas.append(np.mean(v_mtr[segments[i][0]:segments[i][1]] - mV_gps[i]))
        
    x, y = [], []
    for i in range(len(speeds)):
        Y = 0
        for j in range(len(segments)):
            if segments[j][0] <= i < segments[j][1]:
                Y = mV_gps[j]
        y.append(Y)
        x.append(i)
    
    v_new = v_mtr - np.mean(deltas)
    plt.plot(x, v_mtr, x, v_gps, x, y, x, v_new)
    plt.show()
    plt.plot(x[:1400], v_mtr[:1400], x[:1400], v_gps[:1400], x[:1400], y[:1400], x[:1400], v_new[:1400])  
    plt.show()
    sys.exit()
        
    video = cv2.VideoCapture(data)
    times = collections.deque(maxlen=lengh)
    for i in range(1, lengh):
        t0 = time.time()
        print('  saving: {} / {}'.format(i, lengh))
        out = frames[i].copy()
        out = draw.draw_point(out, m)
        out = draw.draw_mahalanobis_ellipse(out, r, m, K, color=draw.red)
        out = draw.draw_mahalanobis_ellipse(out, 3 * r, m, K, color=draw.blue)
        area1 = sf.get_box(frames[i-1], mask)
        area2 = sf.get_box(frames[i], mask)
        pt, st = pai.find_OF_crossing_pt(area1, area2, method='lk')
        r_p = None
        if st:
            if not mask is None:
                pt += mask[0]
                if np.linalg.norm(pt) > np.linalg.norm((img_dim[1], img_dim[0])):
                    pt = (max(img_dim) / 2) * pt / np.linalg.norm(pt)
            r_point_sq = sf.get_mahalanobis_distance_sq(pt, m, K_inv)
            r_p = np.sqrt(r_point_sq)
            out = cv2.line(out, (int(m[0]), int(m[1])), (int(pt[0]), int(pt[1])), color=draw.green)
            out = draw.draw_point(out, pt)
            out = draw.draw_mahalanobis_ellipse(out, r, m, K, color=draw.blue)
            out = draw.draw_mahalanobis_ellipse(out, 3*r, m, K, color=draw.red)
            out = draw.draw_mahalanobis_ellipse(out, np.sqrt(sf.get_mahalanobis_distance_sq(pt, m, K_inv)), m, K, color=draw.gold)
        if status[i] == 1:
            out = draw.draw_arrow(out, m, m - np.array([0, 100]), color=draw.red, thickness=3)
            
        else:
            out = draw.draw_arrow(out, m, m + np.array([100, 0]), color=draw.red, thickness=3)
            out = draw.draw_arrow(out, m, m - np.array([100, 0]), color=draw.red, thickness=3)
        text = dict(text='r(alpha) = ' + str(3 * r), font_scale=1, line_type=2, color=draw.blue)
        out = draw.draw_text(out, (0, 50), **text)
        text = dict(text='r = ' + str(r_p), font_scale=1, line_type=2, color=draw.blue)
        out = draw.draw_text(out, (0, 80), **text)
        text = dict(text='pt = ' + str(pt), font_scale=1, line_type=2, color=draw.blue)
        out = draw.draw_text(out, (0, 120), **text)

        cv2.imwrite('out/_{}.jpg'.format(i), out)
        tk = time.time()
        T = tk - t0
        times.append(T)
        T = (sum(times) / len(times)) * (lengh - i)
        t_min = T // 60
        t_sec = T % 60
        print("Time left: {}min {:.2f}sec".format(int(t_min), t_sec))
#data_c = 'calibration_video.mp4'  # calibration video
#data = 'test1.mp4'                # video
#
#frame_status = collections.deque(maxlen=5000)
#times = collections.deque(maxlen=5000)
#cam = cv2.VideoCapture(data)
#max_frame_index = cam.get(cv2.CAP_PROP_FRAME_COUNT)
#y, x, z = cam.read()[1].shape  # size of frame
#x1, y1, x2, y2 = 0, 0, x, y#x//3, y//4, 2*x//3, y  # points of rectangle that will be analysed in ransac
#pt_rec1 = np.array([x1, y1])
#pt_rec2 = np.array([x2, y2])
#    #  generation of speed array.
#path = 'out/'
#speeds = collections.deque(maxlen=5000)
#file = open('speeds.txt', 'r')
#for line in file:
#    speeds.append(np.float(line.split('|')[1]))
#speeds.append(speeds[len(speeds) - 1])
#speeds.append(speeds[len(speeds) - 1])
#X, Y, _ = sf.read_stat('out/calibration/points.txt')
#r = sf.get_mahalanobis_distance_sq_by_probability(0.8)  # getting mahalanubis radius by probability
#    # generating of new sample by filter, that used as criterion mahalanubis distance
#X, Y = sf.get_subarrays_by_filter(X, Y, 
#                                  sf.indexes_of_points_that_mahalanobis_dist_smaller_than(X, Y, r))
#r = np.sqrt(sf.get_mahalanobis_distance_sq_by_probability(0.95))  # getting mahalanubis radius by probability
#m, K = sf.stat(X, Y)
#
#w, v = np.linalg.eig(K)
#a1x = v[0][0]
#a1y = v[1][0]
#a2x = v[0][1]
#a2y = v[1][1]
#    #if you want to pass a part of the video, where the car is standing on some place:
#_, f1 = cam.read()
#pt1_x, pt1_y = sf.get_min_max_dist_of_ellipse(np.sqrt(r), m, K)
#_, f2 = cam.read()
#i = 0
#
#
##while _ is True:
##    t0 = time.time()
##    area1 = sf.get_box(f1, pt_rec1, pt_rec2)
##    area2 = sf.get_box(f2, pt_rec1, pt_rec2)
##    pt, st = ofl.ransac(area1, area2, method='orb')
##    p = path + '{:05d}.jpg'.format(i)
###    out = f2.copy()
###    out = draw.draw_mahalanobis_ellipse(out, r, m, K, thickness=3)
###    out = draw.draw_line(out, a1y / a1x, m[1] - (a1y / a1x) * m[0], color=draw.blue)
###    out = draw.draw_line(out, a2y / a2x, m[1] - (a2y / a2x) * m[0], color=draw.dark_green)
###    out = draw.draw_point(out, m, color=draw.dark_green, radius=5, thickness=5)
##    
###    text_prop = dict(font_scale=1, line_type=2, color=draw.blue)
###    out = draw.draw_text(out, (0, 50), text='r = {}'.format(np.sqrt(r)), **text_prop)
##    
##    if st:
###        out = cv2.line(out, (int(pt[0]), int(pt[1])), (int(m[0]), int(m[1])), color=draw.green)
###        out = draw.draw_point(out, pt, radius=5, thickness=5)
##        r_point = np.sqrt(sf.get_mahalanobis_distance_sq_by_sample(pt, m, K))
###        out = draw.draw_text(out, (0, 80), text='r_point = ' + str(r_point), **text_prop)
###            out = draw.draw_text(out, (0, 110), text='frame status = 0', **text_prop)  
##        if r_point < r:
##            frame_status.append(1)
###            out = draw.draw_text(out, (0, 110), text='frame status = 1', **text_prop)
##        else:
##            frame_status.append(0)
###            out = draw.draw_text(out, (0, 110), text='frame status = 0', **text_prop)
##    else:
##        frame_status.append(0)
###        out = draw.draw_text(out, (0, 110), text='frame status = 0', **text_prop)
###        text_prop['color'] = draw.red
###        out = draw.draw_text(out, (0, 80), text='r_point = error', **text_prop)
###    text_prop['color'] = draw.blue
###    out = draw.draw_text(out, (0, 140), text='v = {:.2f} km/h'.format(speeds[i + 1]), **text_prop)
###    cv2.imwrite(p, out)
##    f1 = f2
##    i += 1
##    _, f2 = cam.read()
##    tk = time.time()
##    dt = tk - t0
##    times.append(dt)
##    T = (sum(times) / len(times)) * (max_frame_index - i)
##    t_min = T // 60
##    t_sec = T % 60
##    print('\t tau = {}\n\t {:.0f}min {:2.4f}sec'.format(dt, t_min, t_sec))
##    print('COMPLETE: {:.4f}%'.format(100 * i / max_frame_index))
##            
##plt.plot(frame_status)
##plt.show()
##mean = mean_filter(frame_status)
##rmean = np.rint(mean)
##file = open('status.txt', 'w')
##for k in range(i):
##    file.write('{}|{}|{}|{}\n'.format(k, frame_status[k], mean[k], rmean[k]))
##file.close()
#
#file = open('status.txt', 'r')
#rmean = collections.deque(maxlen=5000)
#mean = collections.deque(maxlen=5000)
#data = collections.deque(maxlen=5000)
#for line in file:
#    rmean.append(np.float(line.split('|')[3].strip()))
#    mean.append(np.float(line.split('|')[2].strip()))
#    data.append(np.float(line.split('|')[1].strip()))
##rmean = medfilt(mean, kernel_size=3)
#_rmean = np.rint(sf.mean_filter(np.array(data)))
##for r, d in zip(rmean, data):
##    print(r / 3,d)
##file.close()
##straight_lines = collections.deque(maxlen=5000)
##distances = collections.deque(maxlen=5000)
##o = False
##ik = -1
##jk = -1
##for i in range(len(rmean)):
##    if not o:
##        if rmean[i] == 1:
##            o = True
##            ik = i
##            jk = 0
##    else:
##        if rmean[i] == 1:
##            jk += 1
##        else:
##            o = False
##            if jk > 1:
##                straight_lines.append((ik, ik + jk))
##print(straight_lines)
##frame_rate = 30  # frames per second
##speeds = np.array(speeds)
##for sl in straight_lines:
##    sub_speeds = speeds[sl[0]:sl[1] + 1]
##    mV = (sum(sub_speeds) / len(sub_speeds)) / 3.6
##    S = mV * (sl[1] - sl[0] + 1) / frame_rate
##    distances.append(S)
##print(distances)
##print(len(distances) == len(straight_lines))
#print(_rmean == rmean)
#print(len(rmean))
