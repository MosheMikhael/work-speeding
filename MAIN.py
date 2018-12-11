# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:05:51 2018

@author: moshe.f
"""

import cv2
import draw
import numpy as np
import matplotlib.pyplot as plt
import point_at_infinity as pai
import base_classes as bc
import collections
import service_functions as sf
import time
import sys

def modification_points(pts):
    '''
            To convert array of points to special array of points in 
        special notification in openCV.
    '''
    out = np.zeros((len(pts), 1, 2), dtype=np.float32)
    for i in range(len(pts)):
#        print(i)
        out[i][0][0] = pts[i][0]
        out[i][0][1] = pts[i][1]
    return out

def is_in_side_of_rectangle(pt, pt1, pt2):
    '''
            To check point, is it inside rectangle,
        that top-left corner is pt1 and bottom-right is pt2.
    '''
    if pt1[0] < pt[0] < pt2[0] and pt1[1] < pt[1] < pt2[1]:
        return True
    return False

def get_grid(pt1, pt2, rows, cols):
    '''
            To generate special variable 'grid', that contains matrix
        of characteristic points of rectangles of grid.
    '''
    dx = (pt2[0] - pt1[0]) / cols
    dy = (pt2[1] - pt1[1]) / rows
    grid = np.zeros((rows, cols, 2, 2), dtype = np.int)
    for i in range(rows):
        for j in range(cols):
            grid[i][j][0][0] = pt1[0] + j * dx
            grid[i][j][0][1] = pt1[1] + i * dy
            grid[i][j][1][0] = pt1[0] + (j + 1) * dx
            grid[i][j][1][1] = pt1[1] + (i + 1) * dy
    return grid


def get_ij_in_grid_by_pt(pt, grid):
#    pt0 = grid[0][0][0]
    rows = len(grid)
    cols = len(grid[0])
#    ptk = grid[rows - 1][cols - 1][1]
#    dx = grid[0][0][1][0] - grid[0][0][0][0]
#    dy = grid[0][0][1][1] - grid[0][0][0][1]
#    pt_ = pt - pt0
#    if pt_[0] < 0 or pt_[1] < 0 or pt_[0] > ptk[0] or pt_[1] > ptk[1]:
#        return -1, -1
#    i = int(pt_[0] // dx)
#    j = int(pt_[1] // dy)
    
    for i in range(rows):
        for j in range(cols):
            if is_in_side_of_rectangle(pt, grid[i][j][0], grid[i][j][1]):
                return i, j 
    return -1, -1

def get_angle(pt0_input, pt1_input, pt2_input): # pt1 pt2 m
    pt0 = pt0_input
    pt1 = pt1_input
    pt2 = pt2_input
    
    if is_in_side_of_rectangle(pt2_input, pt0_input, pt1_input):
        pt0 = np.array([pt0_input[0], pt2_input[1]], np.float32)
        
    a = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    b = np.sqrt((pt1[0] - pt0[0]) ** 2 + (pt1[1] - pt0[1]) ** 2)
    c = np.sqrt((pt2[0] - pt0[0]) ** 2 + (pt2[1] - pt0[1]) ** 2)
#    b = np.linalg.norm(pt1 - pt0)
#    c = np.linalg.norm(pt2 - pt0)
    corner = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
    return corner
    
def get_corners(pt1_rec_input, pt2_rec_input, m):
    # GEOMETRY
    pt1_rec = pt1_rec_input
    pt2_rec = pt2_rec_input
    if is_in_side_of_rectangle(m, pt1_rec, pt2_rec):
        pt1_rec[1] = m[1]
    
    A = np.array([pt2_rec[0], pt1_rec[1]], dtype = np.int)
    B = np.array([pt1_rec[0], pt2_rec[1]], dtype = np.int)
    D = np.array([pt2_rec[0], m[1]], dtype = np.int)
    
    v1 = D - m
    v2 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    omega_0 = np.arccos(v1.dot(v2) / (n_v1 * n_v2))
    
    v1 = A - pt2_rec
    v2 = pt2_rec - m
    v3 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega_1 = omega_0 + np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3))
    
    v1 = pt1_rec - A
    v2 = pt1_rec - m
    v3 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega = np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3))
    
    v1 = pt1_rec - B
    v2 = pt1_rec - m
    v3 = B - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega_2 = omega_0 + omega - np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3)) 
    # END OF GEOMETRY
    return omega, omega_0, omega_1, omega_2
    
def gen_points(omega_0, omega_1, omega_2, omega, LINES): 
    d_omega = omega / (LINES - 1)                      # An angle change to adjacent axis
    omega_i = np.zeros(LINES, np.float32)              # An array of angles for each axis
    line_points = np.zeros((LINES, 2), np.float32)
    r_lines = np.zeros(LINES, np.float32)
    N_lines = np.zeros(LINES, np.int)
    points_on_line = np.zeros(LINES, object)
    pts_start = collections.deque()
    for i in range(LINES):
        omega_i[i] = omega_0 + i * d_omega
        r = -1
        if omega_i[i] < omega_1:
            line_points[i][0] = p2_rec[0]
            r = (p2_rec[0] - m[0]) / np.cos(omega_i[i])
            line_points[i][1] = m[1] + r * np.sin(omega_i[i])
        elif omega_i[i] > omega_2:
            line_points[i][0] = p1_rec[0]
            r = (p1_rec[0] - m[0]) / np.cos(omega_i[i])
            line_points[i][1] = m[1] + r * np.sin(omega_i[i])
        else:
            line_points[i][1] = p2_rec[1]
            r = (p2_rec[1] - m[1]) / np.sin(omega_i[i])
            line_points[i][0] = m[0] + r * np.cos(omega_i[i])    
        r_lines[i] = r
#        print(i)
#        print(r)
#        print(int(r / 2))
        N_lines[i] = np.int(r / 2)
        points_on_line[i] = np.zeros((N_lines[i], 2), np.int)
        dr = r / N_lines[i]
        for j in range(N_lines[i]):
            points_on_line[i][j][0] = np.int(m[0] + dr * (j + 1) * np.cos(omega_i[i]))
            points_on_line[i][j][1] = np.int(m[1] + dr * (j + 1) * np.sin(omega_i[i]))
    
    for i in range(len(points_on_line)):
        for j in range(N_lines[i]):
            if is_in_side_of_rectangle(points_on_line[i][j], p1_rec, p2_rec):
                pts_start.append(points_on_line[i][j])
    pts_start = np.array(pts_start, dtype = np.int)
    print(len(pts_start))
    return pts_start
    
def intensity_threshould_filter_of_points(img, points, threshould):
    points_out = collections.deque()
    points_out = []
    bool_points = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    cv2.imwrite('out/g.jpg', gray)
    y, x, _ = img.shape

    for i in range(len(points)):
        for j in range(2, len(points[i])):
            p1 = points[i][j - 2]
            p2 = points[i][j]
            if is_in_side_of_rectangle(p2, (0, 0), (x, y)):

#                print(p1, p2)
                I1 = int(gray[int(p1[1])][int(p1[0])])
                I2 = int(gray[int(p2[1])][int(p2[0])])
                if np.abs(I1 - I2) > threshould:
#                    print(type(p1), p1,  points_out)
#                    p1 = [p1[0], p1[1]]
                    if [p1[0], p1[1]] not in bool_points:
                        points_out.append(p1)
                        bool_points.append([p1[0], p1[1]])
#                    points_out.append(p2)
            else:
                break
    return points_out

def direction_threshould_filter_of_points(pts1, pts2, pt0, threshould):
    pts1_out = collections.deque()
    pts2_out = collections.deque()
    for i in range(len(pts1)):
        alpha_0 = get_direction(pts1[i], m)
        alpha_1 = get_direction(pts2[i], pts1[i])
        if np.abs(alpha_0 - alpha_1) < threshould:
            pts1_out.append(pts1[i])
            pts2_out.append(pts2[i])
    return pts1_out, pts2_out
    
def is_directions_to(pt1, pt2, pt0, alpha_k, alpha_0=0):
#    alpha_00 = get_direction(pt1, pt0)
#    alpha_01 = get_direction(pt2, pt1)
    alpha = kivun_shel_vector(pt1, pt2, pt0)
#    print(alpha * 180 / np.pi)
    if alpha_0 <= alpha < alpha_k:
        return True
    return False
    
    
def get_direction(pt, m):
    x = pt[0] - m[0]
    y = pt[1] - m[1]
    return np.arctan2(y, x)
    G = np.array([pt[0], m[1]], np.int)
    v1 = pt - G
    v2 = G - m
    v3 = pt - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    psi = np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3)) 
    if G[0] < m[0]:
        psi = np.pi - psi
    return psi

def get_points(R, omega_0, LINES, h, pt0 = [0, 0]):
    omega = np.pi - 2 * omega_0
    d_omega = omega / (LINES - 1)
#    N = R / 2
    dr = 2   
    count = 0
#    r = 0
    points = np.zeros(LINES, object)
    for i in range(LINES):
        points[i] = collections.deque()
    omega_i = 0
    for i in range(LINES):
        omega_i = omega_0 + i * d_omega
        r = 0
        while r < R:
            pt = np.zeros(2, np.float32)
            pt[0] = int(r * np.cos(omega_i))
            pt[1] = int(r * np.sin(omega_i))
            if pt[1] <= h:
                points[i].append(pt + pt0)
            count += 1
            r += dr
    return points, count

def kivun_shel_vector(pt1, pt2, m):
    x_0 = m[0]
    y_0 = m[1]
    x = pt1[0]
    y = pt1[1]
    v_x = pt2[0] - pt1[0]
    v_y = pt2[1] - pt1[1]
    cos_t = ((x - x_0) * v_x + (y - y_0) * v_y) / (np.linalg.norm([x - x_0, y - y_0]) * np.linalg.norm([v_x, v_y]))
    return np.arccos(cos_t)

if __name__ == '__main__':    
    video = 'input/calibration_video.mp4'
    #video = 'input/test1.mp4'
    cam = cv2.VideoCapture(video)
    img1 = cam.read()[1]
    img2 = cam.read()[1]
    max_corners = 5000
    N_FRAMES = 700
    times = collections.deque(maxlen=N_FRAMES)
    t_points = bc.Points() # for finding point of infinity
    y, x, z = img1.shape
    img_dim = (y, x, z)
    cap = bc.Capture(video)
    x1, y1, x2, y2 = x // 5, 0.45 * y // 1, 4 * x // 5, 0.85 * y // 1
    p1_rec = np.array([x1, y1])
    p2_rec = np.array([x2, y2])
    mask = [p1_rec, p2_rec]
    rows = 10
    cols = 20
    max_cos = np.cos(20 * np.pi / 180)
    LINES = 20
    grid = get_grid(p1_rec, p2_rec, rows, cols)
    deepness = 10
    intensity_threshould = 20
    arrow_size_factor = 10
    m = np.array([x // 2 - 50,  y * 0.4], np.float32)
    m_new = m
    T = 0
    weight = 0.1
    for I in range(N_FRAMES):
        t0 = time.time()
        img1 = img2
        img2 = cam.read()[1]
        m_points = bc.Points()
    #    omega, omega_0, omega_1, omega_2 = get_corners(p1_rec, p2_rec, m_new)
        #points = gen_points(omega_0, omega_1, omega_2, omega, LINES)
        points, count = get_points(550, 30 * np.pi / 180, LINES, m_new)
        print('Number of points: {}'.format(count))
        
        out = img1.copy()
        for pt_r in points:
            for pt in pt_r:
                out = draw.draw_point(out, np.array([pt[0], pt[1]]), radius=3, color=draw.gold)
        
    #    for i in range(len(points)):
        #    if i % 1000 == 0:
    #        print('  {} / {}'.format(i, len(points)))
    #        for j in range(len(points[i])):
    #            out = draw.draw_point(out, points[i][j], 1)
    #    out = draw.draw_point(out, m, color=draw.dark_green, thickness=3)
    #    cv2.imwrite('out/{}_all.jpg'.format(I), draw.draw_grid(out, grid))
        
    #    out = img1.copy()
        #out = draw.draw_grid(out, grid)
        pts_filtered = intensity_threshould_filter_of_points(img1, points, intensity_threshould)
        print('filtered: {}'.format(len(pts_filtered)))
        for pt in pts_filtered:
            out = draw.draw_point(out, np.array([pt[0], pt[1]]), radius=2, color=draw.blue)
        cv2.imwrite('output/pic/points_all.png', out)
        
    #    for i in range(len(pts_filtered)):
    #        if i % 1000 == 0:
    #            print(' {} / {}'.format(i, len(pts_filtered)))
    #        out = draw.draw_point(out, pts_filtered[i], 3, color = draw.red, thickness=1)
    ##    
    #    out = draw.draw_point(out, m, color=draw.dark_green, thickness=3)
    #    cv2.imwrite('out/{}_filtered.jpg'.format(I), out)
        mod_pts = modification_points(pts_filtered)
        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, mod_pts)
#        out = img1.copy()
        
        for i in range(len(pts1)):
            pt1 = pts1[i]
            pt2 = pts1[i]
            
            out = draw.draw_point(out, np.array([pt1[0], pt1[1]]), radius=1, color=draw.red)
            out = draw.draw_point(out, np.array([pt2[0], pt2[1]]), radius=1, color=draw.red)
        cv2.imwrite('output/pic/points_all.png', out)
        
        sys.exit()
    #    out = draw.draw_point(out, m, color=draw.dark_green, thickness=3)
        good = np.zeros(len(pts1), bool)
        for i in range(len(pts1)):
            if is_directions_to(pts1[i], pts2[i], m, 30 * np.pi / 180):
                good[i] = True
            else:
                out = draw.draw_arrow(out, pts1[i], arrow_size_factor * (pts2[i]-pts1[i]) + pts1[i], color=draw.red)
        for i in range(len(pts1)):
            if good[i]:
                m_points.add_point(pts1[i], pts2[i])
                out = draw.draw_arrow(out, pts1[i], arrow_size_factor * (pts2[i]-pts1[i]) + pts1[i], color=draw.green)
                
        m_points.mark_inlier_all()
        m_points.make_lines()
        pt_m, st = m_points.get_OF_by_max_dens()
    #    for i in range(m_points.get_number_of_points()):
    #        k, b = m_points.get_point_by_id(i).get_line()
    #        out = draw.draw_line(out, k, b)
        out = draw.draw_point(out, m, color=draw.black, thickness=3)
        if st:
            m_new = weight * pt_m + (1 - weight) * m_new
    #        for id in inliers:
    #            k, b = m_points.get_point_by_id(id[1]).get_line()
    #            out = draw.draw_line(out, k, b, color = draw.blue)
            out = draw.draw_point(out, pt_m, color=draw.dark_green, thickness=3)  # current
            
        out = draw.draw_point(out, m_new, color=draw.purple, thickness=4)   # middle
        cv2.imwrite('out/{}_arrows.jpg'.format(I), out)
        
    #    pt, st = m_points.find_OF_crossing_pt(100, 10, 5)
    #    if st:
    #    m = sf.mean_online(m, pt, I)
        
        tk = time.time()
        times.append(tk - t0)
        T = sum(times) / len(times)
        t_min = int(T // 60)
        t_sec = T % 60
        print(' {} | time: {} min {:2.02f} sec'.format(I, t_min, t_sec))
    #    sys.exit('DONE')
    print('DONE')
