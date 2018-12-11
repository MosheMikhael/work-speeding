# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:29:51 2018

@author: moshe.f
"""

import cv2
import draw
import numpy as np
import matplotlib.pyplot as plt
import point_at_infinity as pai
import base_classes as bc
import collections
import time
import sys

def my_mean(m_, data, i):
    return ((i + 1) * m_ + data) / (i + 2)

def mod_points(pts):
    '''
            To convert array of points to special array of points in 
        special notification in openCV.
    '''
    out = np.zeros((len(pts), 1, 2), dtype=np.float32)
    for i in range(len(pts)):
        print(i)
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

def draw_grid(img, grid):
    '''
        Draw grid on image by special variable 'grid'.
    '''
    rows = len(grid)
    cols = len(grid[0])
    out = img.copy()
    for i in range(rows):
        for j in range(cols):
            pt1 = grid[i][j][0]
            pt2 = grid[i][j][1]
            out = draw.draw_rectangle(out, [pt1, pt2], color=draw.cyan)
    return out

def get_angle(pt0, pt1, pt2):
    a = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    b = np.sqrt((pt1[0] - pt0[0]) ** 2 + (pt1[1] - pt0[1]) ** 2)
    c = np.sqrt((pt2[0] - pt0[0]) ** 2 + (pt2[1] - pt0[1]) ** 2)
#    b = np.linalg.norm(pt1 - pt0)
#    c = np.linalg.norm(pt2 - pt0)
    corner = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
    return corner
    
def get_corners(pt1_rec, pt2_rec, m):
    # GEOMETRY
    A = np.array([p2_rec[0], p1_rec[1]], dtype = np.int)
    B = np.array([p1_rec[0], p2_rec[1]], dtype = np.int)
    D = np.array([p2_rec[0], m[1]], dtype = np.int)
    
    v1 = D - m
    v2 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    omega_0 = np.arccos(v1.dot(v2) / (n_v1 * n_v2))
    
    v1 = A - p2_rec
    v2 = p2_rec - m
    v3 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega_1 = omega_0 + np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3))
    
    v1 = p1_rec - A
    v2 = p1_rec - m
    v3 = A - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega = np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3))
    
    v1 = p1_rec - B
    v2 = p1_rec - m
    v3 = B - m
    n_v1 = np.linalg.norm(v1)
    n_v2 = np.linalg.norm(v2)
    n_v3 = np.linalg.norm(v3)
    omega_2 = omega_0 + omega - np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3)) 
    # END OF GEOMETRY
    return omega, omega_0, omega_1, omega_2
    
if __name__ == '__main__':    
    video = 'input/calibration_video.mp4'
    #video = 'input/test1.mp4'
    cam = cv2.VideoCapture(video)
    img1 = cam.read()[1]
    img2 = cam.read()[1]
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_corners = 5000
    N_FRAMES = 700
    times = collections.deque(maxlen=N_FRAMES)
    t_points = bc.Points() # for finding point of infinity
    y, x, z = img1.shape
    img_dim = (y, x, z)
    cap = bc.Capture(video)
    m = np.array([x // 2, y * 0.4], np.float32)
    m_drawed = m
    x1, y1, x2, y2 = x // 5, 0.45 * y // 1, 4 * x // 5, 0.85 * y // 1
    p1_rec = np.array([x1, y1])
    p2_rec = np.array([x2, y2])
    mask = [p1_rec, p2_rec]
    max_cos = np.cos(20 * np.pi / 180)
    LINES = 250
    rows = 10
    cols = 20
    
    
    
    
    
    
    # MAIN SCRIPT
    for I in range(N_FRAMES):
        t0 = time.time()
        
        omega, omega_0, omega_1, omega_2 = get_corners(p1_rec, p2_rec, m)
        line_points = np.zeros((LINES, 2), dtype = np.int) # An array of points at the ends of the axis
        r_lines = np.zeros(LINES, np.float32)              # An array of lengths of axis
        N_lines = np.zeros(LINES, np.int)                  # An array of number of points for each axis
        points_on_line = np.zeros(LINES, object)           # An array of deques with coordinates of points on each axis
        
        d_omega = omega / (LINES - 1)                      # An angle change to adjacent axis
        omega_i = np.zeros(LINES, np.float32)              # An array of angles for each axis
        
    
        deepness = 10
        middle = np.zeros((rows, cols), object)            # An array of memory for all points
        D = np.zeros((rows, cols), np.float32)
        pts_start = collections.deque()
        grid = get_grid(p1_rec, p2_rec, rows, cols)
        for i in range(rows):
            for j in range(cols):
                middle[i][j] = collections.deque(maxlen=deepness)
                u = (grid[i][j][0][0] + grid[i][j][1][0]) / 2 - m[0]
                v = (grid[i][j][0][1] + grid[i][j][1][1]) / 2 - m[1]
                D[i][j] = (v ** 2) * np.sqrt(1 + (u / v) ** 2) / 5000
        
        # INIT POINTS AT AXIS
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
    
        
        
        print('{} / {}'.format(I, N_FRAMES))
        img1 = img2.copy()
        img2 = cam.read()[1]
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        middle_vectors = np.zeros((rows, cols, 3), np.float32)
        middle_vectors_D = np.zeros((rows, cols), np.float32)
        out = img1.copy()    
        out_ = img1.copy()
        out__ = img1.copy()
        points = collections.deque()       # points, that delta intensivity is bigger than 
    
        for i in range(2, len(pts_start)):
            pt1 = (pts_start[i-2][0], pts_start[i-2][1])
            pt2 = (pts_start[i][0], pts_start[i][1])
            out__ = draw.draw_point(out__, pt1, 3, color=draw.black)
            F1 = int(gray[pt1[1]][pt1[0]])
            F2 = int(gray[pt2[1]][pt2[0]])
            d = np.abs(F1 - F2)
            if d > 25:
                points.append(pt1)
    #            print(i)
                out = draw.draw_point(out, pt1, 3)
                out_ = draw.draw_point(out_, pt2, 3, color=draw.cyan)
                
        out = draw_grid(out, grid)
        out_ = draw_grid(out_, grid)
        cv2.imwrite('out/o.jpg', out)
        cv2.imwrite('out/o_.jpg', out_)
        cv2.imwrite('out/o__.jpg', out__)
        
        sys.exit(1)
        
        # optical flow
        points_opt_data = np.zeros((len(points), 1, 2), dtype = np.float32)
        for i in range(len(points)):
            points_opt_data[i][0][0] = points[i][0]
            points_opt_data[i][0][1] = points[i][1]
        pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, points_opt_data)
        # good point filter by direction
        count = 0
        good_points = np.zeros(len(pts1), bool)
        for i in range(len(pts1)):
    ##        v1 = (pts1[i] - m) / np.linalg.norm(pts1[i] - m)
    ##        v2 = (pts2[i] - pts1[i]) / np.linalg.norm(pts1[i] - pts2[i])
    ##        v2 = (pts2[i] - m) / np.linalg.norm(pts2[i] - m)
    #        G = np.array([pts1[i][0], m[1]], np.int)
    #        v1 = pts1[i] - G
    #        v2 = G - m
    #        v3 = pts1[i] - m
    #        n_v1 = np.linalg.norm(v1)
    #        n_v2 = np.linalg.norm(v2)
    #        n_v3 = np.linalg.norm(v3)
    #        fi_1 = np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3))
    #        
    #        pt1 =
    #        if G[0] < m[0]:
    #                fi = np.pi / 2 + fi
    #        _v_ = np.array([np.cos(fi), np.sin(fi)], np.float32) + pts1[i]
    #        cos_ = v1.dot(_v_) / (n_v1 * np.linalg.norm(_v_))
    ##        angle = np.math.atan2(np.linalg.det([_v_,v1]),np.dot(_v_,v1)) * 180 / np.pi
    ##        print(cos_, max_cos)
            
    #        alpha_1 = get_angle(m, pts1[i], np.array([pts1[i][0], m[1]], np.float32))
    #        alpha_2 = get_angle(pts1[i], pts2[i], np.array([pts2[i][0], pts1[i][1]], np.float32))        
            angle = get_angle(pts1[i], pts2[i], m)
    #        print(angle)
            if 180 - angle < 20:
                if pts2[i][1] < m[1]:
                    continue
                v = pts1[i] - pts2[i]
                i_, j_ = get_ij_in_grid_by_pt(pts1[i], grid)
                if i_ == -1 or j_ == -1:
                    continue
                count += 1
                good_points[i] = True
                t_points.add_point(pts1[i], pts2[i])
        print(' {} arrows / {} points'.format(count, len(points)))
        for i in range(len(pts1)):
            if good_points[i]:
                out = draw.draw_arrow(out, (pts1[i][0], pts1[i][1]), (pts2[i][0], pts2[i][1]), thickness=1)
    #            out = draw.draw_point(out, pts1[i], radius=2, color=draw.black)
    #            out = draw.draw_point(out, pts2[i], radius=2, color=draw.black)
                for l in range(rows):
                    for k in range(cols):
                        if is_in_side_of_rectangle(pts1[i], grid[l][k][0], grid[l][k][1]):
                            middle_vectors[l][k][0] += pts2[i][0] - pts1[i][0]
                            middle_vectors[l][k][1] += pts2[i][1] - pts1[i][1]
                            middle_vectors[l][k][2] += 1
            else:
                out = draw.draw_arrow(out, (pts1[i][0], pts1[i][1]), (pts2[i][0], pts2[i][1]), thickness=1, color=draw.red  )
    
        for i in range(rows):
            for j in range(cols):
                if middle_vectors[i][j][2] == 0:
                    middle[i][j].append(np.zeros(2, np.float32))
                else:
                    middle[i][j].append(middle_vectors[i][j][0:2] / middle_vectors[i][j][2])
                
    
    #    FIELD = np.zeros((rows, cols), np.float32)
        norm_f = collections.deque(maxlen=rows*cols)
        good_points_counter = 0
        bad_points_counter = 0
        
        for i in range(rows):
            for j in range(cols):            
                # corner for each middle point in rectangles
                pt1 = (grid[i][j][0] + grid[i][j][1]) / 2
                G = np.array([pt1[0], m[1]], np.int)
                v1 = pt1 - G
                v2 = G - m
                v3 = pt1 - m
                n_v1 = np.linalg.norm(v1)
                n_v2 = np.linalg.norm(v2)
                n_v3 = np.linalg.norm(v3)
                psi = np.arccos((n_v2 ** 2 + n_v3 ** 2 - n_v1 ** 2) / (2 * n_v2 * n_v3)) 
                if G[0] < m[0]:
                    psi = np.pi - psi
                pt2_D =  D[i][j] * np.array([np.cos(psi), np.sin(psi)], np.float32)
                out = draw.draw_arrow(out, pt1, pt1 + pt2_D, color=draw.white, thickness=1)
    #            pt2 = np.zeros(2, np.float32)
    #            if len(middle[i][j]) > 0:
                pt2 = np.mean(middle[i][j], axis=0)
                good_points_counter += 1
                norm_f.append(np.linalg.norm(pt2) / D[i][j])  
                if np.linalg.norm(pt2) / D[i][j] == 0:
                    bad_points_counter += 1
                    out = draw.draw_point(out, pt1, 4, draw.orange, 3)
                out = draw.draw_arrow(out, pt1, pt1 + pt2, color = draw.purple, thickness=1)
        out = draw.draw_point(out, m_drawed, 5, draw.blue, 5)
    #    out = draw.draw_rectangle(out, mask, color=draw.cyan)
        out = draw_grid(out, grid)
        
        mu = np.median(norm_f)
        sigma = np.std(norm_f)
        n, bins, patches = plt.hist(norm_f, 10, density=True, facecolor='g', alpha=0.75)
        filtered_norm_f = collections.deque(maxlen = rows*cols)
        for i in range(len(norm_f)):
            if mu - sigma <= norm_f[i] < mu + sigma:
                filtered_norm_f.append(norm_f[i])
        n, bins, patches = plt.hist(filtered_norm_f, 10, density=True, facecolor='r', alpha=0.75)
    
    #   drawing histogram
        plt.savefig('out/fig.png')
        fig = cv2.imread('out/fig.png')
        f_y, f_x, _ = fig.shape
        out[0:f_y, x - f_x:x] = fig
        
        speed = np.median(filtered_norm_f)
        print('median is {}'.format(speed))
        out = draw.draw_text(out, (50, 50), 'Speed: {:02.06f}'.format(speed), font_scale=2, line_type=3)
        out = draw.draw_text(out, (50, 110), ' Error: {:02.02f} %'.format(bad_points_counter / good_points_counter * 100), font_scale=2, line_type=3)
        m_new = t_points.find_OF_crossing_pt(num_cycles = 40, num_rnd_lines = 30,
                         delta = 5, trace = False, path = 'output/', img = img2)
        if m_new[1]:
            m_drawed = my_mean(m_drawed, m_new[0], I)
            m = m_drawed
        cv2.imwrite('out/out{}.jpg'.format(I), out)
        tk = time.time()
        T = tk - t0
        times.append(T)
        T = (sum(times) / len(times))
        T_min = int(T // 60)
        T_sec = T % 60
        print('  assessment of time per frame: {} min {:02.2f} sec'.format(T_min, T_sec))
    
    print('> > > DONE !!! < < <')
