# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:41:22 2018

@author: moshe.f
"""

def log_read(adr):
    lines = []
    with open(adr, 'r') as file:
        lines = file.readlines()
    frame_id = []
    time_of_start_for_frame_analysis = []
    time_per_frame = []
    time_by_gps = []
    loc = []
    model = []
    sALG = []
    sGPS = []
    sAME = []
    pai = []
    for line in lines:
        if 'PROGRAM WAS STARTED AT ' in line:
            start = line.split(' ')[4:]
            start = [start[0], start[1], int(start[2]), [int(start[3].split(':')[0]), int(start[3].split(':')[1]), int(start[3].split(':')[2])], int(start[4])]
        elif 'frame #' in line:
            frame_id.append(int(line.split('#')[1][:-2]))
        elif 'current time:' in line:
            item = line.split(':')[1:]
            time_of_start_for_frame_analysis.append([item[0].split(' ')[0], item[0].split(' ')[1], item[0].split(' ')[2], int(item[0].split(' ')[3]), [int(item[0].split(' ')[4]), int(item[1]), int(item[2].split(' ')[0])], int(item[2].split(' ')[1])])
        elif 'time per frame:' in line:
            time_per_frame.append(float(line.split(':')[1]))
        elif 'time by GPS:' in line:
            time_by_gps.append(float(line.split(':')[1]))
        elif 'loc=' in line:
            loc.append((float(line.split('=')[1].split(':')[0][1:-2].split(',')[0]), float(line.split('=')[1].split(':')[0][1:-2].split(',')[1])))
        elif '(Ax,Ay,Az,B)=' in line:
            item = line.split('=')[1]
            if 'None' in item:
                model.append((None, None, None, None))
            else:
                item = item[1:-2].split(',')
                model.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
        elif 'sALG' in line:
            if 'None' in line:
                sALG.append(None)
            else:
                sALG.append(float(line.split('=')[1]))
        elif 'sAM7' in line:
            if 'None' in line:
                sAME.append(None)
            else:
                sAME.append(float(line.split('=')[1]))
        elif 'sGPS' in line:
            if 'None' in line:
                sGPS.append(None)
            else:
                sGPS.append(float(line.split('=')[1]))
        elif 'pai' in line:
            if 'None' in line:
                pai.append(None)
            else:
                pai.append((float(line.split('=')[1].split(':')[0][1:-2].split(',')[0]), float(line.split('=')[1].split(':')[0][1:-2].split(',')[1])))
    
    return dict(start_time=start, frame_ids=frame_id, time_of_start_frame=time_of_start_for_frame_analysis, time_per_frame=time_per_frame, time_by_gps=time_by_gps, 
                loc=loc, model=model, sALG=sALG, sGPS = sGPS, sAME = sAME)
        
log = log_read('output/log.txt')
#print('out', out)
#print(out['sALG'][2892])