# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:58:31 2018

@author: moshe.f
"""
import collections
from enum import Enum
import numpy as np

class Rotation(Enum):
    Left = 0
    Left_Straight = 1
    Straight = 2
    Right_Straight = 3
    Right = 4
    #def __getitem__(self):
    #    return self.value
    def __radd__(self, other):
        return other + self.value
    
class mideller:
    def __init__(self, size=3):
        self.size = size
        self.data = collections.deque(maxlen=size)
    
    def push_prev(self):
        s = np.rint(sum(self.data) / len(self.data))
        self.push(int(s))
        
    def push(self, frame_stat):
        if np.isnan(frame_stat):
            self.push_prev()
            return
        self.data.append(frame_stat)
    
    def get_middle(self):
        #if len(self.data) == self.size:
#            print('\t {} | {} | {} | mid: {} ~ {}'.format(self.data[0], self.data[1], self.data[2], sum(self.data) / len(self.data), np.rint(sum(self.data) / len(self.data))))
        return sum(self.data) / len(self.data)    
    