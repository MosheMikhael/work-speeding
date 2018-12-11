# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:47:28 2018

@author: moshe.f
"""
import collections
import numpy as np

def maximum_figure(u, speed, B_factor, max_OF, H, L):
    if speed is None:
        speed = 50
    if np.abs(speed) < 1:
        speed = 1
    return np.sqrt(np.sqrt((u ** 4) / 4 + (H * L * B_factor * max_OF/ speed) ** 2) - (u ** 2) / 2)

def vector_flow_forward(u, v, L, H=1):
    tmp = v / (H * L)
    return -u * tmp, -v * tmp

def vector_flow_rotation_x(u, v, L):
    tmp = v / L
    return -u * tmp, L + v * tmp

def vector_flow_rotation_y(u, v, L):
    tmp = u / L
    return L + u * tmp, v * tmp

def vector_flow_rotation_z(u, v):
    return -u, v

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

def get_field_vector(pt, A_x, A_y, A_z, B, L, pai = np.array([0, 0])):
    u = pt[0] - pai[0]
    v = pt[1] - pai[1]
    du_x, dv_x = vector_flow_rotation_x(u, v, L)
    du_y, dv_y = vector_flow_rotation_y(u, v, L)
    du_z, dv_z = vector_flow_rotation_z(u, v)
    du_f, dv_f = vector_flow_forward(u, v, L)
    vx = np.array([du_x, dv_x])
    vy = np.array([du_y, dv_y])
    vz = np.array([du_z, dv_z])
    vf = np.array([du_f, dv_f])
    return A_x * vx + A_y * vy + A_z * vz + B * vf

def build_model(pts1, pts2, pai, L, bad_number=10):
    '''
        This function returns parameters of model.
    :param pts1: array of first point in optical flow.
    :param pts2: array of second point in optical flow.
    :param pai: point of infinity.
    :param L: focus lenght.
    :return: sum of residuals / number of points.
    '''
    n_pts = len(pts1)
    if n_pts <= bad_number:
        print('Bad set')
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
        du_x, dv_x = vector_flow_rotation_x(u, v, L)
        du_y, dv_y = vector_flow_rotation_y(u, v, L)
        du_z, dv_z = vector_flow_rotation_z(u, v)
        du_f, dv_f = vector_flow_forward(u, v, L)
        
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
        du_x, dv_x = vector_flow_rotation_x(u, v, L)
        du_y, dv_y = vector_flow_rotation_y(u, v, L)
        du_z, dv_z = vector_flow_rotation_z(u, v)
        du_f, dv_f = vector_flow_forward(u, v, L)
        
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
        M[1][1] += F_y[k][0] ** 2 + F_y[k][1] ** 2
        M[1][2] += F_y[k][0] * F_z[k][0] + F_y[k][1] * F_z[k][1]
        M[2][2] += F_z[k][0] ** 2 + F_z[k][1] ** 2
        
        V[0] += F_o[k][0] * F_x[k][0] + F_o[k][1] * F_x[k][1]
        V[1] += F_o[k][0] * F_y[k][0] + F_o[k][1] * F_y[k][1]
        V[2] += F_o[k][0] * F_z[k][0] + F_o[k][1] * F_z[k][1]
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
        A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o = build_model(pts1_clean, pts2_clean, pt_current, L) # current model
        residuals_clean = get_residuals(A_x, A_y, A_z, B, F_x, F_y, F_z, F_f, F_o, pts1_clean, pts2_clean) # current residuals
        sum_current[i] = sum_of_residuals(residuals_clean) 
        x_ = pt_current[0]
        y_ = pt_current[1]
        M[i][0] = x_  ** 2
        M[i][1] = x_
        M[i][2] = y_  ** 2
        M[i][3] = y_
        M[i][4] = 1
    return M, sum_current
