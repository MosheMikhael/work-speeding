# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:16:34 2018

@author: moshe.f
"""
import numpy as np
import collections
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
v0 = 10
omega = 1 / 10
N = 4133
X_out, Y_out = collections.deque(maxlen=N), collections.deque(maxlen=N)
file = open('speeds.txt', 'w')
v = -1
X_in = np.array([0, 283, 490, 1000, 2000, 2250, 2600, 4133])
Y_in = np.array([2, 0, 20, 50, 60, 40, 50, 30])
f = UnivariateSpline(X_in, Y_in)

for i in range(N):       
    v = f(i)        
    line = '{}|{}\n'.format(i, v)
    X_out.append(i)
    Y_out.append(v)
    file.write(line)
file.close()
plt.plot(X_out, Y_out)
plt.show()
