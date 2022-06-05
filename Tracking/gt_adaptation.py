
import os
import numpy as np
from skimage import io
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import sys
from PIL import Image

import pandas as pd
'''
STEP 1: INITALITATION
STEP 2: MAKE DICTIONARY WITH GROUND TRUTH FOR 4 CAMERA
STEP 3: PRINT IN OUTPUT FILE 
STEP 4: MAKE OUTPUTFILE WITH ALL DETECTIONS

'''

folder_name = 'S02'
'STEP 1'

string_output = 'data/gt/'+folder_name+'-train/'

if folder_name =="S01":
    path_to_train = 'data/AIC21_Track3_MTMCTracking/train/S01'
else:
    path_to_train = 'data/AIC21_Track3_MTMCTracking/validation/S02'

path_to_num_frames = 'data/AIC21_Track3_MTMCTracking/cam_framenum/'+folder_name+'.txt'
path_to_timestamp = 'data/AIC21_Track3_MTMCTracking/cam_timestamp/'+folder_name+'.txt'
pattern = os.path.join(path_to_train,'*', 'gt', 'gt.txt')

if not os.path.exists(path_to_train):
    sys.exit("ERROR: path to train does not exist")


if not os.path.exists(string_output):
    os.makedirs(string_output)

framenumbers  =  np.loadtxt(path_to_num_frames,dtype=int,usecols = [1]) 
number_of_cameras = len(framenumbers)
max_frame = max(framenumbers)


# MAKE A LIST WITH ALL trackers
all_seq_det_tracker = [0]*number_of_cameras
for i,seq_dets_path_tracker in enumerate(glob.glob(pattern)):
    #print(seq_dets_path_tracker)
    all_seq_det_tracker[i] = np.loadtxt(seq_dets_path_tracker, delimiter=',')


#make dictonnary for timestamps
timestamps= {}
with open(path_to_timestamp) as f:
    for line in f:
        (key, val) = line.split()
        timestamps[key] = np.float(val)
max_ts = max(timestamps.values())
for t in timestamps:
    timestamps[t]=  round((max_ts - timestamps[t])*10)




# intialise ground_truth_tracker
ground_truth_tracker= {}
for seq_dets_fn in glob.glob(pattern):
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 

    ground_truth_tracker[seq]= []


'STEP 2'
for frame in range(min(framenumbers)):
    frame += 1
    all_trackers_at_time_frame = {}

    #prend tous les trackers time frame
    for i,seq_dets_fn in enumerate(glob.glob(pattern)):
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 

        if seq != 'c005':
            frame = frame + timestamps[seq]
            dets_tracker = all_seq_det_tracker[i][all_seq_det_tracker[i][:, 0]==frame] # frame, id, x1, y1, w, h ,1, R,G,B
            all_trackers_at_time_frame[seq] = dets_tracker
            frame = frame - timestamps[seq]
    
    #Make a dictonary that counts number of time each id is present
    global_id_dic = {}
    for seq in all_trackers_at_time_frame.keys():
        all_keys = global_id_dic.keys()
        all_trackers = all_trackers_at_time_frame[seq]
        for id in all_trackers[:,1]:
            if id in all_keys:
                global_id_dic[id] = global_id_dic[id]+1
            else: 
                global_id_dic[id] = 1

    #only take ids that are 2 time presents
    for seq in all_trackers_at_time_frame.keys():
        all_trackers = all_trackers_at_time_frame[seq]
        for i,id in enumerate(all_trackers[:,1]):
            if global_id_dic[id] > 1:
                ground_truth_tracker[seq].append(all_trackers[i])
         

for seq in ground_truth_tracker.keys():
    if not os.path.exists(os.path.join(string_output, '%s/gt'%(seq))):
        os.makedirs(os.path.join(string_output, '%s/gt'%(seq)))
        
    with open(os.path.join(string_output, '%s/gt/gt.txt'%(seq)),'w') as out_file:

        for d in ground_truth_tracker[seq]:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[0],d[1],d[2],d[3],d[4],d[5]),file=out_file)

'STEP4'
if not os.path.exists(os.path.join(string_output, 'all_cam/gt')):
    os.makedirs(os.path.join(string_output , 'all_cam/gt'))

with open(os.path.join(string_output , 'all_cam/gt/gt.txt'),'w') as out_file:
    number_of_frames =0
    for seq in ground_truth_tracker.keys():
        if seq != 'c005':
            print(seq)
            for d in ground_truth_tracker[seq]:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(number_of_frames + d[0],d[1],d[2],d[3],d[4],d[5]),file=out_file)
            number_of_frames = number_of_frames + max_frame


print("gt created for 4 cameras and car visible at minimum 2")