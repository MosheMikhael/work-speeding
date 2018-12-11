# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:43:26 2018

@author: moshe.f
"""
import cv2
adr = 'input/1750be37-5499-4c6e-9421-9bb15b277a94.mp4'
cam = cv2.VideoCapture(adr)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
o, img = cam.read()
y, x, z = img.shape
fx = 0.5
fy = 0.5
out = cv2.VideoWriter('out' + '.avi', fourcc, 20.0, (int(x * fx), int(y * fy)))

counter = 0
while o:
    if counter % 500 == 0:
        print(counter)
    counter += 1
    img = cv2.resize(img, (0,0), fx=fx, fy=fy)
    out.write(img)
    o, img = cam.read()

out.release()
print('done')