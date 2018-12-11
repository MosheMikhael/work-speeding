# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 13:35:16 2018

@author: moshe.f
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

X = np.zeros(10, dtype=np.float32)
Y = np.zeros(10, dtype=np.float32)
Y_ = np.zeros(10, dtype=np.float32)
for i in range(10):
    X[i] = i
    Y[i] = i ** 2 + np.random.standard_normal()
    
p2 = scipy.polyfit(X, Y, 2)
for i in range(10):
    Y_[i] = p2[0] * X[i] ** 2 + p2[1] * X[i] + p2[2]
    
plt.plot(X, Y, X, Y_)