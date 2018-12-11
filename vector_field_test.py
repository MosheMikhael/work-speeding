# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:51:33 2018

@author: moshe.f
"""

import numpy as np
import model as mdl
L = 10_000
N = 20
max_x = 256
max_y = 256

A_x = 0.0345
A_y = -0.04
A_z = -0.05
B = -0.164
pts1 = [np.array([np.random.randint(max_x), np.random.randint(max_y)]) for i in range(N)]
pts2 = [pt + mdl.get_field_vector(pt, A_x, A_y, A_z, B, L) for pt in pts1]
vZero = np.array([0, 0])
A_x_, A_y_, A_z_, B_, _, _, _, _, _ = mdl.build_model(pts1, pts2, vZero, L)
print('old =', A_x, A_y, A_z, B)
print('new =', A_x_, A_y_, A_z_, B_)
err = np.sqrt((A_x - A_x_) ** 2 + (A_y - A_y_) ** 2 + (A_z - A_z_) ** 2 + (B - B_) ** 2)
print('error =', err)