# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:18:00 2018

@author: moshe.f
"""

import numpy as np
import sys
def haversine(lon1, lat1, lon2, lat2, dem='km'):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    r = -1
    if dem == 'km':
        r = 6371.3
    elif dem == 'ml':
        r = 3956
    # convert decimal degrees to radians 
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1) 
    lon2 = np.deg2rad(lon2) 
    lat2 = np.deg2rad(lat2)

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    angle = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(angle)) 
    return c * r
#sys.exit()
if __name__ == '__main__':
    filename = 'input/1750be37-5499-4c6e-9421-9bb15b277a94.txt'
    lat = []
    lon = []
    distances = []
    time = []
    H = -1
    M = -1
    S = -1
    with open(filename, 'r') as file:
        last_lat = -1
        last_lon = -1
        for line in file:
            if '<trkpt'  in line:
                line_items = line.split(' ')
                lat_item = float(line_items[4].split('"')[1])
                lon_item = float(line_items[5].split('"')[1])
                distance = 0
                if last_lat != -1 and last_lon != -1:
                    distance = haversine(last_lon, last_lat, lon_item, lat_item)
                print('coord =  {} | {} => distance = {}'.format(lat_item, lon_item, distance), flush = True)
    #            sys.exit()
                last_lat = lat_item
                last_lon = lon_item
                distances.append(distance)
                lat.append(lat_item)
                lon.append(lon_item)
            if '<time>' in line:
                time_items = line.split('T')[1].split('Z')[0].split(':')
                h = np.int(time_items[0])
                m = np.int(time_items[1])
                s = np.int(time_items[2])
                if H == -1:
                    H = h
                    M = m
                    S = s
                delta_H = h - H
                delta_M = m - M
                delta_S = s - S
                if delta_S < 0:
                    delta_M -= 1
                    delta_S += 60
                if delta_M < 0:
                    delta_H -= 1
                    delta_M += 60
                index = delta_S + 60 * delta_M + 3600 * delta_H                
                print('time = {}'.format([h, m, s]))
                time.append([h, m, s])
                
    with open('input/parsed-1750be37-5499-4c6e-9421-9bb15b277a94.txt', 'w') as file:
        for i in range(len(lat)):
            line = '{:02d}:{:02d}:{:02d}|{:.07f}|{:.07f}|{}\n'.format(time[i + 1][0], time[i + 1][1], time[i + 1][2], lat[i], lon[i], distances[i])
            file.write(line)