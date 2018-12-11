# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:10:08 2018

Using optical flow to find speed.

@author: moshe.f
"""
import numpy as np
import cv2
import service_functions as sf
import point_at_infinity as pai
import draw
import matplotlib.pyplot as plt
import base_classes as bc
import time
import sys
import collections
import scipy
from enum import Enum


class Algorithm(Enum):
    Simple = 0
    Modificated = 1

class Field:
    def __init__(self, x=None, y=None, eps=None, good=None):
        if not eps is None:
            self.set_field(x, y, eps)
        else:
            self.x = x
            self.y = y
            self.eps = eps
            self.good = good
    def get_num_of_grid_points(self):
        return len(self.eps)
    def set_field(self, x, y, eps):
        self.x = x
        self.y = y
        self.eps = eps  
        self.good = np.zeros(len(eps), dtype=bool)
        for i in range(len(eps)):
            if self.eps[i] != -1:
                self.good[i] = True
            if np.isnan(self.eps[i]):
                self.eps[i] = -1
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_eps(self):
        return self.eps
    def get_avr_eps(self):
        return avr(self.eps)
    
    def get_speed(self, template):
        return self._get_speed_with_weights1(template)
    
    def __get_speed_without_weights(self, template):
        speed = 0
        count = 0
        for i in range(self.get_num_of_grid_points()):
            if self.good[i]:
                count += 1
                speed += np.sqrt(self.x[i] ** 2 + self.y[i] ** 2) 
        return speed / count
    def _get_speed_with_weights1(self, template):
        '''
            This is default!!!
        '''
        norm_coef = 0
        speed = 0
        for i in range(self.get_num_of_grid_points()):
            if self.good[i]:
                coef = 1 / (self.eps[i] ** 2 + template.eps[i] ** 2)
                norm_coef += coef
                speed += np.sqrt(self.x[i] ** 2 + self.y[i] ** 2) * coef
        speed /= norm_coef
        return speed
    def __get_speed_with_weights2(self, template):
        norm_coef = 0
        speed = 0
        for i in range(self.get_num_of_grid_points()):
            if self.good[i]:
                coef = 1 / np.sqrt((self.eps[i] ** 2 + template.eps[i] ** 2))
                norm_coef += coef
                speed += np.sqrt(self.x[i] ** 2 + self.y[i] ** 2) * coef
        speed /= norm_coef
        return speed
    def get_statistica(self, m, k):
        '''
            Field quality.
        '''
        roh = 0
        for i in range(1, m - 1):
            for j in range(1, k - 1):
                ind = i * k + j
                roh = roh + np.linalg.norm(np.array([self.x[ind], self.y[ind]], dtype=np.float32) - np.array([self.x[ind+1], self.y[ind]], dtype=np.float32)) + \
                       np.linalg.norm(np.array([self.x[ind], self.y[ind]], dtype=np.float32) - np.array([self.x[ind], self.y[ind+1]], dtype=np.float32)) + \
                       np.linalg.norm(np.array([self.x[ind], self.y[ind]], dtype=np.float32) - np.array([self.x[ind-1], self.y[ind]], dtype=np.float32)) + \
                       np.linalg.norm(np.array([self.x[ind], self.y[ind]], dtype=np.float32) - np.array([self.x[ind], self.y[ind-1]], dtype=np.float32))
#                print(' |{}| - |{}| = {}'.format(np.array([self.x[i], self.y[i]]), np.array([self.x[i+1], self.y[j]]), roh))
#        print(' {} / {} = {}'.format(roh, 4 * (m - 1) * (k - 1), roh / 4 * (m - 1) * (k - 1)))
        roh /= 4 * (m - 2) * (k - 2)
        return roh
        
    def save(self, filename):
        save_field(filename, self.x, self.y, self.eps)
    def load(self, filename):
        x, y, e = load_field(filename)
        self.set_field(x, y, e)
    def draw(self, frame, mask, points, output, scale=10, pref=''):
        out = draw.draw_rectangle(frame, mask, draw.cyan)
        avr_eps = self.get_avr_eps()
        for i in range(N):
            t0 = time.time()
            
            color = draw.green
            if self.eps[i] < avr_eps:
                color = draw.red
            if self.eps[i] == -1:
                color = draw.cyan
            pt1 = points[i][0]
            
            pt2 = pt1 + scale * np.array([self.x[i], self.y[i]])
            out = draw.draw_point(out, pt1, 1, draw.blue)
            out = draw.draw_point(out, pt2, 1, draw.blue)
            out = draw.draw_arrow(out, pt1, pt2, color)
            
            tk = time.time()
            times.append(tk - t0)
            T = (sum(times) / len(times)) * (N - i)
            t_min = np.int(T // 60)
            t_sec = T % 60
            print(pref + 'drawing: {} / {}\t {} min {:.2f} sec'.format(i, N, t_min, t_sec))
        cv2.imwrite(output, out)    

        
def id_to_ij(id, col):
    '''
        Convert index from vector of points to coordinates in grid of points.
    '''
    return id % col, id // col

def ij_to_id(i, j, col):
    '''
        Convert coordinates of points in grid to index in vector of points.
    '''
    return j * col + i


def avr(arr, v=-1):
    '''
        Average value of array, with out v-containing.
    '''
    out = 0
    counter = 0
    for i in range(len(arr)):
        if not arr[i] == v:
            counter += 1
            out += arr[i]
    return out / counter

def get_grid(frame, mask, m, k):
    '''
        Getting array of points, that have a coordinates of grid.
    '''
    area1 = sf.get_box(frame, mask)
    y, x, z = area1.shape
    N = m * k
    # greed points
    points = np.zeros((N, 1, 2), dtype = np.float32)
    Ax = mask[0][0]
    Ay = mask[0][1]
    Bx = mask[1][0]
    By = mask[1][1]    
    y, x, z = frame.shape
    y0 = Ay + 0.02 * (By - Ay)
    x0 = Ax + 0.02 * (Bx - Ax)
    yk = By - 0.02 * (By - Ay)
    xk = Bx - 0.02 * (Bx - Ax)
    dy = (yk - y0) / m
    dx = (xk - x0) / k
    for i in range(m*k):
        points[i][0][0] = x0 + dx * (i % k)
        points[i][0][1] = y0 + dy * (i // k)
    return points

def get_optical_flow_field_lk(frame2, frame1, points):
    '''
        Optical flow between two frames.
    '''
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
    points, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), points, None, **lk_params)     
    return points, st

def get_mod_optical_flow_field_lk(frame1, frame2, points, dx=3, dy=3, nx=3, ny=3, pref=''):
    ex = np.array([dx, 0], dtype=np.float32)
    ey = np.array([0, dy], dtype=np.float32)
    N = len(points)
    st = np.zeros(N, dtype=np.int)
    out_points = np.zeros((N, 1, 2), dtype=np.float32)
    
    for itt in range(N):
        if itt % (N // 5) == 0:
            print('    mean of points: {} / {}'.format(itt, N))
    
        tmp_points = np.zeros((nx * ny, 1, 2), dtype=np.float32)
        # ----------
#        out = frame2.copy()
        # ----------
        for i in range(ny): # creating points in window 
            for j in range(nx):
                tmp_points[i * nx + j][0] = points[itt][0] + (i - ny // 2) * ey + (j - nx // 2) * ex
                
        tmp_opt_points, tmp_st = get_optical_flow_field_lk(frame1, frame2, tmp_points)
        tmp_mean = np.zeros((2), dtype=np.float32)
        count = 0
        for i in range(nx * ny):
            
            # ----------
#            out = draw.draw_point(out, tmp_points[i][0], 1, draw.blue)
            # ----------
            if tmp_st[i] == 1:
                count += 1
                # ----------
#                out = draw.draw_arrow(out, tmp_points[i][0], 20*(tmp_opt_points[i][0]-tmp_points[i][0]) + tmp_points[i][0], draw.green)
                # ----------
#                print('  dx = {} | {} {}'.format(tmp_opt_points[i][0][0] - tmp_points[i][0][0], tmp_opt_points[i][0][0], tmp_points[i][0][0]))
#                print('  dy = {} | {} {}'.format(tmp_opt_points[i][0][1] - tmp_points[i][0][1], tmp_opt_points[i][0][1], tmp_points[i][0][1]))
                tmp_mean[0] += tmp_opt_points[i][0][0]
                tmp_mean[1] += tmp_opt_points[i][0][1] 
        if count != 0:
            tmp_mean /= count
            tmp_mean -= points[itt][0]
            st[itt] = 1
            out_points[itt][0][0] = tmp_mean[0] + points[itt][0][0]
            out_points[itt][0][1] = tmp_mean[1] + points[itt][0][1]
#        print('{} / {}'.format(itt, N))
#        print(' count = {}'.format(count))
#        print(' mean = {} | {}'.format(tmp_mean, np.linalg.norm(tmp_mean)))
#        out = draw.draw_text(out, (50, 80), 'count = {} mean = {} | {} st = {}'.format(count, tmp_mean, np.linalg.norm(tmp_mean), tmp_st), draw.blue)
        
#        out = draw.draw_text(out, (50, 110), 'points = {}'.format(tmp_opt_points), draw.blue)
#        out = draw.draw_text(out, (50, 140), 'norm = {}'.format(np.linalg.norm(tmp_mean)), draw.blue)
        
#        out = draw.draw_arrow(out, points[itt][0], tmp_mean + points[itt][0], draw.cyan, 3)
#        out = draw.draw_point(out, points[itt][0], 2)
#        cv2.imwrite('out/task4/{}_{}.jpg'.format(pref, itt), out)
    return out_points, st
        
    
def get_field(cam, points, n_frames, min_num_points=5, algorithm=Algorithm.Modificated ,trace=True, pref=''):
    N = len(points)
    
    sumX = np.zeros(N, dtype=np.float32)
    sumY = np.zeros(N, dtype=np.float32)
    sumX2 = np.zeros(N, dtype=np.float32)
    sumY2 = np.zeros(N, dtype=np.float32)
    Num = np.zeros(N, dtype=np.int)
    
    avr_x = np.zeros(N, dtype=np.float32)
    avr_y = np.zeros(N, dtype=np.float32)
    std_x2 = np.zeros(N, dtype=np.float32)
    std_y2 = np.zeros(N, dtype=np.float32)
    eps = np.zeros(N, dtype=np.float32)
    
    _, frame1 = cam.read()
    _, frame2 = cam.read()
    
    # frame stack
    frames = collections.deque(maxlen=n_frames)
    frames.append(frame1)
    frames.append(frame2)
    #
    
    times = collections.deque()
    for itt in range(n_frames):
        st, points_new = -1, -1
        t0 = time.time()
        if algorithm == Algorithm.Simple:
            print('Simple')
            points_new, st = get_optical_flow_field_lk(frame1, frame2, points)
        elif algorithm == Algorithm.Modificated:
            print('Modificated')
            points_new, st = get_mod_optical_flow_field_lk(frame1, frame2, points)
        tk = time.time()
        times.append(tk - t0)
        
        for i in range(N):
            if st[i] == 1:
                addX = points_new[i][0][0] - points[i][0][0]
                addY = points_new[i][0][1] - points[i][0][1]
                Num[i] += 1
                sumX[i] += addX
                sumY[i] += addY
                sumX2[i] += addX ** 2
                sumY2[i] += addY ** 2
        frame1 = frame2
        _, frame2 = cam.read()
        
        #
        frames.append(frame2)
        #
        
        if not _:
            sys.exit('Video is ended!')
        if trace:
            T = (sum(times) / len(times)) * (n_frames - itt)
            t_sec = T % 60
            t_hou = np.int(T // 60 // 60)
            t_min = np.int(T % 3600 // 60)
            print(pref + 'optical flow: {} / {}\t times: {} h {} min {} sec'.format(itt, n_frames, t_hou, t_min, t_sec))
    for itt in range(N):
        if Num[itt] < min_num_points:
            eps[itt] = -1
        else:
            avr_x[itt] = sumX[itt] / Num[itt]
            avr_y[itt] = sumY[itt] / Num[itt]
            std_x2[itt] = sumX2[itt] / Num[itt] - avr_x[itt] ** 2
            std_y2[itt] = sumY2[itt] / Num[itt] - avr_y[itt] ** 2
            
            eps[itt] = np.sqrt(std_x2[itt] + std_y2[itt])
        if trace:
            print(pref + 'calculating (stage I): {} / {}'.format(itt, N))
    avr_eps = sum(eps) / len(eps)
#    count = 0 
    ind = collections.deque(maxlen=N)
    m = np.int(np.sqrt(len(points)))
    for i in range(len(eps)):
        r, c = id_to_ij(i, m)
        if eps[i] > avr_eps and r != 0 and c != 0 and r != m and c != m and i < len(points) - m:
#            count += 1
            ind.append(i)
    print(ind)
    pt1 = np.zeros((len(ind), 2), np.float32)
    pt2 = np.zeros((len(ind), 2), np.float32)
    
    vector_x = np.zeros(len(ind), np.float32)
    vector_y = np.zeros(len(ind), np.float32)
    vector_c = np.zeros(len(ind), np.float32)
    
    for i in range(len(ind)):
        pt1[i][0] = points[ind[i] - m - 1][0][0]
        pt1[i][1] = points[ind[i] - m - 1][0][1]
        pt2[i][0] = points[ind[i] + m + 1][0][0]
        pt2[i][1] = points[ind[i] + m + 1][0][1]

    print('stage II')
    for itt in range(1, n_frames):
        print('frame: {} / {}'.format(itt, n_frames))
        for i in range(len(ind)):
            print(' area: {} / {}'.format(i, len(ind)))
            area1 = frames[itt-1][np.int(pt1[i][1]):np.int(pt2[i][1]), np.int(pt1[i][0]):np.int(pt2[i][0])]
            area2 = frames[itt][np.int(pt1[i][1]):np.int(pt2[i][1]), np.int(pt1[i][0]):np.int(pt2[i][0])]
            pts1, pts2 = pai.find_opt_flow(area1, area2, method='lk', prop=0.3)
            print(type(pts1))
            if type(pts1) is type(None):
                pts1 = np.array([-1, -1], dtype=np.float32)
                pts2 = np.array([-1, -1], dtype=np.float32)
            vectors = pts2 - pts1
            
            out = area1.copy()
            for j in range(len(pts1)):
                out = draw.draw_arrow(out, pts1[j], pts2[j])
            cv2.imwrite('out/{}_{}.jpg'.format(itt, i), out)
            
            for _i in range(len(vectors)):
                vector_x[i] += vectors[_i][0]
                vector_y[i] += vectors[_i][1]
                vector_c[i] += 1
        vector_x[i] /= vector_c[i]
        vector_y[i] /= vector_c[i]
    
    for i in range(len(ind)):
        avr_x[ind[i]] = vector_x[i]
        avr_y[ind[i]] = vector_y[i]
#    print(ind)
#    print(len(ind))
#    print(avr_eps)
#    print(n_frames)
#    print(len(frames))
    
    return Field(avr_x, avr_y, eps)

def normalization(field, template, pref=''):    
    norm_x = np.array(field.x, dtype=np.float32)
    norm_y = np.array(field.y, dtype=np.float32)
    N = len(template.eps)
    good = np.zeros(N, dtype=bool)
    for i in range(N):
        print(pref + 'normalization: {} / {}'.format(i, N))
        if template.eps[i] != -1 and field.eps[i] != -1:
            good[i] = True
            norm = np.sqrt(template.x[i] ** 2 + template.y[i] ** 2)
            norm_x[i] /= norm
            norm_y[i] /= norm
    return Field(norm_x, norm_y, field.eps, good)

#def get_speed(template_x, template_y, template_eps, x, y, eps):
    

def save_field(filename, avr_x, avr_y, eps):
    N = len(eps)
    with open(filename, 'w') as file:
        for i in range(N):
            line = '{}|{}|{}|{}\n'.format(i, avr_x[i], avr_y[i], eps[i])
            file.write(line)

def load_field(filename):
    with open(filename, 'r') as file:
        x = collections.deque()
        y = collections.deque()
        e = collections.deque()
        for line in file:
            substr = line.split('|')
            x.append(np.float32(substr[1]))
            y.append(np.float32(substr[2]))
            e.append(np.float32(substr[3]))
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(e, dtype=np.float32)
    print('Error in file reading.')

def get_point_of_infinity(field, points, delta=5, ransac_number=30, num_rnd_lines=20):
    pts = bc.Points()
    points_of_cross = collections.deque(maxlen=ransac_number)
    dist = collections.deque(maxlen=ransac_number)
    for i in range(len(points)):
        pts.add_point(points[i][0], points[i][0] + np.array([field.x[i], field.y[i]], dtype=np.float32))
    pts.make_lines()
    for cycle in range(ransac_number):
        subset = pts.get_subset_of_rnd_lines(num_rnd_lines)
        pt = pts.point_of_crossing(subset)
        if not np.isnan(pt[0]):
            points_of_cross.append(pt)
            dist.append(pts.get_sum_dist(pt, subset))
        if pts.get_number_of_inliers() <= num_rnd_lines:
            break
    if len(dist) == 0:
        return [np.NaN, np.NaN], False
    id_temp_point = list(dist).index(min(dist))
    temp_point = points_of_cross[id_temp_point]
    # Marking outliers
    pts.check_for_lines_in_interesting_area(temp_point, delta)
    inliers = pts.get_indexes_of_inliers()
    pt = pts.point_of_crossing(inliers)
    # if wasn't found point of infinity (some error in numpy.linalg.lstsq)
    if np.isnan(pt[0]):
        return pt, False
    return pt, True

def courner_by_delta(delta, N, foi=20):
    theta = (delta / N) * foi
    return theta

def our_fun(p, r, theta):
    i0, ik = 1, 2 + 1
    j0, jk = 0, 2 + 1
    out = 0
    for i in range(i0, ik):
        for j in range(j0, jk):
            out += (r ** i) * (p['alpha'][i - 1][j] * np.cos(j * theta) + p['betta'][i - 1][j] * np.sin(j * theta))
    return out
    

def euclidean_to_polar(x, y, x0, y0):
    theta = np.arctan2(x - x0, y - y0)
    R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return R, theta
    
def polar_to_euclidean(r, theta, x0, y0):
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    return x, y

def get_functia_by_field(field, points, frame):
    pt0, st = get_point_of_infinity(field, points, delta=5, ransac_number=30, num_rnd_lines=5)
    
    if not st:
        sys.exit('ERROR, i do not see a point of infinity.')
    f = draw.draw_point(frame, pt0, color=draw.cyan, thickness=3)
    alpha = np.zeros((2, 3), dtype = np.float32)
    betta = np.zeros((2, 3), dtype = np.float32)
    N = field.get_num_of_grid_points()
    X = collections.deque(maxlen=N)
    Y = collections.deque(maxlen=N)
    X_ = collections.deque(maxlen=N)
    Y_ = collections.deque(maxlen=N)
    
    R = collections.deque(maxlen=N)
    theta = collections.deque(maxlen=N)
    V = collections.deque(maxlen=N)
    ind = collections.deque(maxlen=N)

    for i in range(N):
        e1 = (pt0 - points[i][0]) / np.linalg.norm(pt0 - points[i][0])
        e2 = np.array([field.x[i], field.y[i]]) / np.linalg.norm(np.array([field.x[i], field.y[i]]))
        a1 = np.arctan2(e1[0], e1[1]) * 180 / np.pi
        a2 = np.arctan2(e2[0], e2[1]) * 180 / np.pi
        da = np.abs(a1 - a2)
#        print(da)
        if field.eps[i] == -1 or da > 10:
            continue
        ind.append(i)
        X.append(np.linalg.norm(points[i][0] - pt0))
        Y.append(np.sqrt(field.x[i] ** 2 + field.y[i] ** 2))
        f = draw.draw_point(f, points[i][0], color=draw.gold, radius=3, thickness=3)
        r, t = euclidean_to_polar(points[i][0][0], points[i][0][1], pt0[0], pt0[1])
        R.append(r)
        theta.append(t)
        V.append(np.sqrt(field.x[i] ** 2 + field.y[i] ** 2))
        
    for i in range(1, 2 + 1):
        for j in range(0, 2 + 1):
            top_a = 0
            top_b = 0
            bot_a = 0
            bot_b = 0
            for k in range(len(ind)):
                top_a += V[k] * (R[k] ** i) * np.cos(j * theta[k])
                bot_a += (R[k] ** (2*i)) * (np.cos(j * theta[k]) ** 2)
                top_b += V[k] * (R[k] ** i) * np.sin(j * theta[k])
                bot_b += (R[k] ** (2*i)) * (np.sin(j * theta[k]) ** 2)
                
        alpha[i - 1][j] = top_a / bot_a
        betta[i - 1][j] = top_b / bot_b
#    new_points = np.zeros((N, 2), dtype=np.float32)
    par = {'alpha':alpha, 'betta':betta}
#    for i in range(len(ind)):
#        data_x = points[ind[i]][0][0]
#        data_y = points[ind[i]][0][1]
#        data_r, data_t = euclidean_to_polar(data_x, data_y, pt0[0], pt0[1])
#        X_.append(np.linalg.norm(np.array([data_x, data_y]) - pt0))
#        data = our_fun(par, data_r, data_t)
#        Y_.append(data)
#        e1 = (pt0 - points[ind[i]][0]) / np.linalg.norm(pt0 - points[ind[i]][0])
#        f = draw.draw_arrow(f, points[ind[i]][0], points[ind[i]][0] + 10 * data * e1, color=draw.cyan) 
    for i in range(N):
        data_x = points[i][0][0]
        data_y = points[i][0][1]
        data_r, data_t = euclidean_to_polar(data_x, data_y, pt0[0], pt0[1])
        X_.append(np.linalg.norm(np.array([data_x, data_y]) - pt0))
        data = our_fun(par, data_r, data_t)
        Y_.append(data)
        e1 = (pt0 - points[i][0]) / np.linalg.norm(pt0 - points[i][0])
        f = draw.draw_arrow(f, points[i][0], points[i][0] + 10 * data * e1, color=draw.cyan) 

    print(par)
#        f = draw.draw_arrow(f, points[i][0], 50 * e1 + points[i][0], thickness=1)
#        f = draw.draw_point(f, points[i][0], radius=4, color=draw.gold, thickness=4)
#    p = scipy.polyfit(X, Y, 2)
#    
#    x = np.zeros(N, dtype=np.float32)
#    y = np.zeros(N, dtype=np.float32)
#    for i in range(N):
#        pass
##        y[i] = p2[0] * X[i] ** 2 + p2[1] * X[i] + p2[2]
#    plt.plot(X, Y, 'o')
    plt.plot(X, Y, 'o', X_, Y_, 'o')
    cv2.imwrite('out/task3/out.jpg', f)
    return par


if __name__ == '__main__':
    def cos_alpha(x0, y0, x1, y1):
        print(x0, y0, x1, y1)
        return (x0 * x1 + y0 * y1) / (np.sqrt(x0 ** 2 + y0 ** 2) * np.sqrt(x1 ** 2 + y1 ** 2))
    
    video1 = 'input/calibration_video.mp4'
    video2 = 'input/test1.mp4'
    video70 = 'out/ff.avi'
    output = 'out/_.jpg'
    cam = cv2.VideoCapture(video1)
    _, frame = cam.read()
    print(_)
    m = 15 # rows
    k = 15 # col
    N = m * k
    alg = Algorithm.Modificated
    times = collections.deque(maxlen=N)
    y, x, z = frame.shape
    x1, y1, x2, y2 = x // 5, y // 5, 4 * x // 5, 0.92 * y // 1
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    mask = [p1, p2]
    points = get_grid(frame, mask, m, k)
    parts = 20

    t = get_field(cam, points, 10, algorithm=alg, pref=' t :')
#    t.save('out/task4/t.txt')
#    t.draw(frame, mask, points, 'out/task4/t.jpg')
    x = np.arange(m)
    y = np.zeros(k, np.float32)
    nrm = np.zeros((m, k), np.float32)
    
    for i in range(m):
        for j in range(k):
            nrm[i][j] = np.linalg.norm(np.array([t.x[i*m+j], t.y[i*m+j]]))
            
    for i in range(m):
        print('row = ' + str(i))
        for j in range(k):
            y[j] = nrm[i][j]
        plt.plot(x, y)
        plt.show()
#    f = np.zeros(parts, object)
#    nf = np.zeros(parts, object)
#    t = Field()
#    t.load('out/task3/t.txt')
#    t.draw(frame, mask, points, 'out/task3/t.jpg')

#    pt_t, st_t = get_point_of_infinity(t, points, delta=5, ransac_number=30, num_rnd_lines=3)
#    if not st_t:
#        sys.exit('ERROR')
#    else:
#        print('All good')
#    pts = np.zeros((parts,2), np.float32) - 1
#    delta = np.zeros(parts, np.float32) - 1    
#    f = np.zeros(parts, object)
#    nf = np.zeros(parts, object)
#    for i in range(parts):
#        f[i] = Field()
#        f[i].load('out/task3/f_{}.txt'.format(i))
#        nf[i] = Field()
#        nf[i].load('out/task3/nf_{}.txt'.format(i))
#    
#    t.draw(frame, mask, points, 'out/task3/t.jpg')
#    ff = cv2.imread('out/task3/t.jpg')    
#    par = get_functia_by_field(t, points, ff)
#    cam = cv2.VideoCapture(video2)
#    for i in range(parts):
#        pref = ' {} : '.format(i)
#        f[i] = get_field(cam, points, 70, algorithm=alg, pref=pref)
#        _, frame = cam.read()
#        f[i].save('out/task3/f_{}.txt'.format(i))
#        nf[i] = normalization(f[i], t, pref)
#        nf[i].save('out/task3/nf_{}.txt'.format(i))
#        nf[i].draw(frame, mask, points, 'out/task3/pic_nf_{}.jpg'.format(i))
#        f[i].draw(frame, mask, points, 'out/task3/pic_f_{}.jpg'.format(i))
#        pt, st = get_point_of_infinity(f[i], points)
#        out_p = cv2.imread('out/task3/pic_f_{}.jpg'.format(i))
#        if st:
#            pts[i] = pt
#            delta[i] = np.linalg.norm(pt_t - pt)
#            out_p = draw.draw_arrow(out_p, pt_t, pt)
#            out_p = draw.draw_point(out_p, pt, radius=5,color=draw.purple, thickness=4)
#        cv2.imwrite('out/task3/pic_poi_{}.jpg'.format(i), out_p)            
#    cam = cv2.VideoCapture(video)
#    for i in range(9):
#        f[i] = get_field(cam, points, 70, algorithm=alg, pref=' {} : '.format(i))
#        nf[i] = normalization(f[i], t, pref='nf[{}]: '.format(i))
        
#    speeds = np.zeros(parts, np.float32)
#    status = np.zeros(parts, np.float32)
#    
#    for i in range(parts):
#        speeds[i] = nf[i].get_speed(t)
#        status[i] = nf[i].get_statistica(m, k)
    
        
#    with open('out/task3/data.txt', 'w') as f:
#        for i in range(parts):
#            f.write('{}|{}|{}|{}\n'.format(i, speeds[i], status[i], delta[i]))

#    x = np.arange(10)
#    e = np.zeros(10)
#    e += 1
#    plt.plot(x, speeds_without, x, speeds_with_1, x, speeds_with_2, x, e)
            

    
    print(' > > >     DONE ! ! !     < < < ')