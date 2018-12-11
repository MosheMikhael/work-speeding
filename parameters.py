# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:05:54 2018

@author: moshe.f
"""
import numpy as np
import logreader 
import matplotlib.pyplot as plt
import sys
#sys.exit()
#dT = -12
#factor = 0.9 => -30 * 0.9 = -27
#data_file_x1 = open('input/data/tGPS.txt', 'r')
#data_file_y1 = open('input/data/sGPS.txt', 'r')
#data_file_x2 = open('input/data/tB.txt', 'r')
#data_file_y2 = open('input/data/sB.txt', 'r')
#X2 = np.float64(data_file_x1.readlines())
#Y2 = np.float64(data_file_y1.readlines())
#X1 = np.float64(data_file_x2.readlines())
#Y1 = np.float64(data_file_y2.readlines())
#
#ind_start = 5150 + 10 #214.4 sec
#ind_end   = 8875 + 20 #396.8 sec
#
#X2 = X2[ind_start:ind_end + 1]
#Y2 = Y2[ind_start:ind_end + 1]
logs = logreader.log_read('log.txt')
X1 = logs['time_per_frame']
Y1 = logs['sAM7']
X2 = logs['time_by_gps']
Y2 = logs['sGPS']
x = range(0, len(X1))
#f, a = plt.subplots()
#a.set_title('Two dates.')
#a.axis([210, 400, -5, 120])  
#a.set_ylim(bottom=-10, top=120)
#a.plot(X2, Y2, 'ro', color='blue', markersize=3, label='GPS')
#a.plot(X1, Y1, 'ro', color='red', markersize=2, label='-30B')
#f.legend()
N1 = len(X1)
N2 = len(X2) 

def search_par(factor, T=None):
    binSize = 5
    out = []
    T1 = -50
    T2 = 50
    if not T is None:
        T1 = T
        T2 = T + 1
    for dt in range(T1, T2):
        bins = {}
        for i in range(N1):
#            binNum = round((X1[i] + dt) / binSize)
            
            binNum = round((x[i] + dt) / binSize)
            if binNum in bins:
                if Y1[i] is not None:
                    bins[binNum].append(Y1[i] * factor)
            else:
                if Y1[i] is not None:
                    bins[binNum] = [Y1[i] * factor]
                
        for i in range(N2):
            binNum = round(X2[i] / binSize)
            
            binNum = round((x[i] + dt) / binSize)
            if binNum in bins:
                if Y2[i] is not None:
                    bins[binNum].append(Y2[i])
            else:
                if Y2[i] is not None:
                    bins[binNum] = [Y2[i]]
        Sum = 0
#        print(dt, len(bins), bins.keys())
        for item in bins:
            Sum += np.std(bins[item])
        out.append(Sum / len(bins))
#    k = list(bins.keys())
#    print(bins[k[0]])    
    return out
sys.exit()
outs = []
for factor in np.arange(-2, 2, 0.01):
    print(factor)
    out = search_par(factor)
    outs.append(out[0])
#factor = 1
#out = search_par(factor)
#print(out.index(min(out)))
        