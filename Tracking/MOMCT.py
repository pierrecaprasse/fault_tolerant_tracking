
from __future__ import print_function
import os
from unittest import skip
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import sys
from Tracking.from_video_to_frame import trans_video_to_frame
np.random.seed(0)
from PIL import Image
from time import sleep
import pandas as pd
from Tracking.utils import apply_clustering, get_dominant_color
from Tracking.sort import Sort
import random
#import haversine as hs


#tracker :  frame, id, x,y,w,h, color, -1,-1-1

def sort_monoc_nocolor_proj(args):
  # Si images sont pas accessible, transformer le flux vidéo en image
  # SI DISPLAY OFF:
  #     Appliquer le sort alogithm sur chaque camera
  #     Ecrire dans le fichier OUTPUT1 le resultats des bbx: ID et 
  # SI DISPLAY ON:
  #     Montré caméra par caméra le mono tracking + occupancy map
  #     +meme que DISPLAY OFF
  # OUTPUT output_gt_no_color 
  
  '''ERROR CASES:
  CASE 1: delete 2 detections of car on cam2 of local id 34
  CASE 2:
  CASE 3:
  CASE 4:
  
  '''
  error_case = args.error_case
  error_per  = args.error_per
  error_cam  = args.error_cam
  print(" ")
  print("SORT OFFLINE NO COLOR")
  print(" ")
  # all train

  folder_name =args.folder_name
  output_file = 'data/single_trackers/'+folder_name+'/output_gt_no_color'
  if folder_name =="S01":
    path_to_train = 'data/AIC21_Track3_MTMCTracking/train/S01'
  else:
    path_to_train = 'data/AIC21_Track3_MTMCTracking/validation/S02'



  wait_3secondes = False
  display = args.display1

  phase = args.phase #phase = train
  num_cam = 0
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display

  # plt.ion() # interactive mode
  # fig = plt.figure()

  if(display):
    plt.ion() # interactive mode
    fig = plt.figure()
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122)
    ax2.set_xlim([0, 5000])
    ax2.set_ylim([0, 5000])



  if not os.path.exists(output_file):
    os.makedirs(output_file)

  det_pattern = args.detections
  if det_pattern == "gt":
    pattern = os.path.join(path_to_train,'*', 'gt', 'gt.txt')
  elif det_pattern == "masc":
    pattern = os.path.join(path_to_train,'*', 'det', 'det_mask_rcnn.txt') 
  elif det_pattern == "yolo":
    pattern = os.path.join(path_to_train,'*', 'det', 'det_yolo3.txt') # pattern :data//AIC21_Track3_MTMC_Tracking/train/s1/*/det/det_mask_rcnn.txt
  



  if not os.path.exists(path_to_train):
    sys.exit("ERROR: path to train does not exist")



  # BOUCLE pour chaque vidéo: create images from video
  print("transforming video to frame from camera ")
  for seq_dets_fn in glob.glob(pattern):
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 
    path_to_video = os.path.join(path_to_train,seq)
    #print("transforming video to frame from camera ",seq)
    trans_video_to_frame(path_to_video)


  print("pars arg max age", args.max_age)
  print("pars arg min hits", args.min_hits)
  # BOUCLE pour chaque vidéo: create sort-tracker
  for seq_dets_fn in glob.glob(pattern):
    

    #print('seq_dets_fn: ',seq_dets_fn) # data/AIC21_Track3_MTMC_Tracking/train/S01/c002/det/det_mask_rcnn.txt ( juste l'endroit * qui va changer)
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker

    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  #list de list du fichier seq_dets_fn (gt.txt)

      

    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]  #just le nom de la sequence exemple ADL_Rundle ou c001
    
    
    '''ERROR CASES:'''
    #case 1

    if error_case ==1:
      percentage_to_delete = error_per/100
      np.random.shuffle(seq_dets)
      n_fist_ind = list(range(0,int(len(seq_dets)*percentage_to_delete)))
      seq_dets = np.delete(seq_dets,n_fist_ind,axis=0)
    
    #case 2
    if error_case ==2:
      if error_cam == 0:
          percentage_to_delete = error_per/100
          np.random.shuffle(seq_dets)
          n_fist_ind = list(range(0,int(len(seq_dets)*percentage_to_delete)))
          seq_dets = np.delete(seq_dets,n_fist_ind,axis=0)
      else:
        seq_to_delete = "c00" + str(error_cam)
        if seq == seq_to_delete:
          percentage_to_delete = error_per/100
          np.random.shuffle(seq_dets)
          n_fist_ind = list(range(0,int(len(seq_dets)*percentage_to_delete)))
          seq_dets = np.delete(seq_dets,n_fist_ind,axis=0)
      

    if seq == "c005":
      skip

    else:
      num_cam +=1
      path_to_cal =  os.path.join(path_to_train,seq,'calibration.txt')
      calibration_doc =  pd.read_csv(path_to_cal,  sep=":|,", header=None, engine='python')
      calibration_string = calibration_doc[1][0][1:]
      calibration_matrix = [    line.split(' ') for line in calibration_string.split(';')]
      calibration_matrix = [[float(cell) for cell in line] for line in calibration_matrix]


      with open(os.path.join(output_file, '%s.txt'%(seq)),'w') as out_file:
        print("Processing %s."%(seq))
        #BOUCLE POUR CHAQUE FRAME
        for frame in range(int(seq_dets[:,0].max())):
          if frame%500==0:
            print("Processing frame %s."%(frame))

          frame += 1 #detection and frame numbers begin at 1
          dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
          dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
          total_frames += 1
    
          if(display): #affiche image +partie 2
            fn = os.path.join(path_to_train, seq, 'images', 'frame%d.jpg'%(frame-1))
            im =io.imread(fn)
            a,b,c = im.shape
            ax1.imshow(im)
            plt.title(seq + ' Tracked Targets')

          start_time = time.time()

          trackers = mot_tracker.update(dets) #IMPORTANT A CHEC

          cycle_time = time.time() - start_time
          total_time += cycle_time

          for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file) #ecrit dans le output file!!
            if(display): #Afficher les trackers!!! 
              pass
              d = d.astype(np.int32) # forme:[x1 y1 x2 y2   id] : [222 179 256 288   2]
              ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
              #ax1.plot((d[0]+ d[2])/2, (d[1]+ d[3])/2, color ='tab:blue')
              middlepoint = [(d[0]+ d[2])/2,(d[1]+ d[3])/2]

              ax1.plot((d[0]+ d[2])/2, (d[1]+ d[3])/2, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
              newpoint = np.dot( np.linalg.inv(calibration_matrix),[(d[0]+ d[2])/2,(d[1]+ d[3])/2,1])
              ax2.plot(newpoint[0]/newpoint[2], newpoint[1]/newpoint[2], marker="o", markersize=5, markeredgecolor="red", markerfacecolor=colours[d[4]%32,:])
              #print('middlepoint:',middlepoint)
              #newpoint = np.dot(calibration_matrix,[(d[0]+ d[2])/2,(d[1]+ d[3])/2,1])
          
              #newpoint = np.dot(np.transpose(calibration_matrix),[(d[0]+ d[2])/2,(d[1]+ d[3])/2,1])

              #newpoint = np.dot( np.linalg.inv(np.transpose(calibration_matrix)),[(d[0]+ d[2])/(2*a),(d[1]+ d[3])/(2*b),1])
  
              #print('newpoint:',newpoint)
              # print('newpoint:', newpoint[0]/ newpoint[2], newpoint[1]/newpoint[2],newpoint[2])
          if(display):
            if wait_3secondes:
              time.sleep(3)               

          # Affiche le carré de detection original
          if False:
            for d2 in dets:
              if(display): #Afficher les trackers!!! 
                d2 = d2[0:4]
                d2 = d2.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d2[0],d2[1]),d2[2]-d2[0],d2[3]-d2[1],fill=False,lw=3))


          if(display): #afficher image
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()
            ax2.cla()

    #print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, frame, frame / total_time))     
  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time)) 
  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / (total_time*num_cam)))
  if(display):
    print("Note: to get real runtime results run without the option: --display")


def display_offline(args):

    '''
    Description: 
      Etape 1: initialise all global variables (frames with good timestamp, homography matrices etc)
      Etape 2:
      for each frame: 
        A: load all detections at time frame t 
        B: apply clustering:
              out: -matches,global_trackers
        C: plot occupancy map (with matches)
        D: plot :
            for each seq: image and bounding boxes
    '''
    print(" ")
    print(" CLUSTERING ")
    print(" ")

    print("Feedback: ",args.feedback)
    ''' ETAPE 1'''
    # ALL GLOBAL VARIABLES AND PATH TO DATA #

    display = args.display2 
    global_count = 0
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) #used only for display

    folder_name =args.folder_name
    if folder_name =="S01":
      path_to_train = 'data/AIC21_Track3_MTMCTracking/train/S01'
    else:
      path_to_train = 'data/AIC21_Track3_MTMCTracking/validation/S02'


    #string_input = 'output_final_c_gt/*'
    #string_input = 'output_gt_def_cam2/*'

    #input_file  = 'data/single_trackers/output_gt_no_color_cam2_fail/*'
    input_file  = 'data/single_trackers/'+folder_name+'/output_gt_no_color/*'
    file_output = 'data/trackers/' + folder_name+'-train/'+folder_name+'/data'


    if not os.path.exists(file_output):
      os.makedirs(file_output)
    #string_output = 'output_final_c/*'

    path_to_num_frames = 'data/AIC21_Track3_MTMCTracking/cam_framenum/'+folder_name+'.txt'
    path_to_timestamp = 'data/AIC21_Track3_MTMCTracking/cam_timestamp/'+folder_name+'.txt'

    framenumbers  =  np.loadtxt(path_to_num_frames,dtype=int,usecols = [1]) 
    number_of_cameras = len(framenumbers)
    max_frame = max(framenumbers)
   
    #make dictonnary for timestamps
    timestamps= {}
    with open(path_to_timestamp) as f:
      for line in f:
          (key, val) = line.split()
          timestamps[key] = np.float(val)
    max_ts = max(timestamps.values())
  
    for t in timestamps:
      timestamps[t]=  round((max_ts - timestamps[t])*10)


    # MAKE A LIST WITH ALL trackers
    all_seq_det_tracker = [0]*number_of_cameras
    for i,seq_dets_path_tracker in enumerate(glob.glob(input_file)):
      #print(seq_dets_path_tracker)
      all_seq_det_tracker[i] = np.loadtxt(seq_dets_path_tracker, delimiter=',')






        #all_seq_det_tracker[i] = np.loadtxt(seq_dets_path_tracker, delimiter=',')

  
    #Make a dictonnary with ALL HOMOGRPAHY MATRIX
    all_cal_matrices = {}
    for i,seq_dets_path_tracker in enumerate(glob.glob(input_file)):
          seq = seq_dets_path_tracker[len(seq_dets_path_tracker)-8:-4]
          path_to_cal =  os.path.join(path_to_train,seq,'calibration.txt')
          calibration_doc =  pd.read_csv(path_to_cal,  sep=":|,", header=None, engine='python')
          calibration_string = calibration_doc[1][0][1:]
          calibration_matrix = [    line.split(' ') for line in calibration_string.split(';')]
          calibration_matrix = [[float(cell) for cell in line] for line in calibration_matrix]
          all_cal_matrices[seq] = calibration_matrix

    axes = []
    if(display):
      plt.ion()
      fig = plt.figure()
      ax1 = plt.subplot2grid((2, 3), (0, 0)) 
      ax2 = plt.subplot2grid((2, 3), (0, 2))
      ax3 = plt.subplot2grid((2, 3), (1, 0))
      ax4 = plt.subplot2grid((2, 3), (1, 2))

      ax5 = plt.subplot2grid((2, 3), (0, 1),rowspan=2)
      plt.tight_layout()

      ax5.xaxis.set_visible(False)
      ax5.yaxis.set_visible(False)
      ax5.title.set_text("Occupancy map")
      axes = [ax1,ax2,ax3,ax4,ax5]

    dic_axes ={}
    count = 0
    for seq_dets_path_tracker in glob.glob(input_file):
      seq = seq_dets_path_tracker[len(seq_dets_path_tracker)-8:-4]
      if seq =="c005":
        skip
      else:
        dic_axes[seq] =count
        count +=1

    matches =np.array([[0,0,0]])
    matches = np.delete(matches, (0), axis=0)
    all_trackers_old_time_frame = {}

    feedback_count = [0,0]
    for frame in range(min(framenumbers)):
      if frame%200==0:
        print("Processing frame %s."%(frame))
      frame += 1
      all_trackers_at_time_frame = {}

      
      #prend tous les trackers time frame
      for i,seq_dets_path_tracker in enumerate(glob.glob(input_file)):
        seq = seq_dets_path_tracker[len(seq_dets_path_tracker)-8:-4]
        if seq != 'c005':
        
          frame = frame + timestamps[seq]
          # if seq == 'c002' and (frame == 748 or frame == 749):
          #   print("special_case")
          dets_tracker = all_seq_det_tracker[i][all_seq_det_tracker[i][:, 0]==frame] # frame, id, x1, y1, w, h ,1, R,G,B
          all_trackers_at_time_frame[seq] = dets_tracker
          frame = frame - timestamps[seq]


      '''Apply CLUSTERING ''' 
      feedback = args.feedback


      start_time = time.time()
      Global_trackers,matches,feedback_trackers,feedback_count= apply_clustering(args,all_trackers_at_time_frame,all_trackers_old_time_frame,all_cal_matrices, matches,global_count,feedback_count,feedback)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      if len(feedback_trackers) != 0:
        #feedback accepted
        #print(feedback_trackers)
        for tracker in feedback_trackers:
          seq = tracker[1]
          i = dic_axes[seq]

          #plot middle point of tracker
          if(display):
            axes[i].plot(tracker[3], tracker[4], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
            ax5.plot(-tracker[5], tracker[6], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
            



      #find new count of id
      if matches.size != 0:
        global_count = max( max( list(map(int, matches[:,0]))) +1,global_count)
        


      # print("sorted mathces:")
      # print(matches)
      # print(" ")
      # print("global trackers")
      # print(Global_trackers)
      #### Apply CLUSTERING ####   

      matches = matches[matches[:,0].argsort()] 
   
      if frame ==1:
        for seq, dets_tracker in Global_trackers.items():
          with open(os.path.join(file_output, '%s.txt'%(seq)),'w') as out_file:
            for d in dets_tracker:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[0],d[1],d[2],d[3],d[4],d[5]),file=out_file)
      else:
        for seq, dets_tracker in Global_trackers.items():
          with open(os.path.join(file_output, '%s.txt'%(seq)),'a') as out_file:
            for d in dets_tracker:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[0],d[1],d[2],d[3],d[4],d[5]),file=out_file)

      if (display): 

        ax5.set_xlim([-42.526, -42.5254])
        ax5.set_ylim([ -90.724,-90.723])
        old_id = -1


        # afficher projection points + occupancy map
        for globalid,seq,seq_id in matches: 
        
            i = dic_axes[seq]

            tracker = all_trackers_at_time_frame[seq][all_trackers_at_time_frame[seq][:,1]==float(seq_id)]
            calibration_matrix = all_cal_matrices[seq]

            d = tracker[0,2:6]
            d = d.astype(np.int32)       
            d[2] = d[2] +d[0]
            d[3] = d[3] +d[1]

            #plot middle point of tracker
            axes[i].plot((d[0]+ d[2])/2, (d[1]+ d[3])/2, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
            
            #plot the projection of the middle point on the occupancy map 
            newpoint = np.dot( np.linalg.inv(calibration_matrix),[(d[0]+ d[2])/2,(d[1]+ d[3])/2,1])
            ax5.plot(- newpoint[0]/newpoint[2], newpoint[1]/newpoint[2], marker="o", markersize=5, markeredgecolor="red", markerfacecolor=colours[int(globalid)%32,:])
            #time.sleep(1) 
             

            #compute estimated projection point 
            if old_id != globalid:
              if old_id != -1:
                #plot the estimated projection point
                ax5.plot(estimated_position[0], estimated_position[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor=colours[int(old_id)%32,:])
                ax5.text(estimated_position[0], estimated_position[1], int(globalid)-1,fontsize=14, color=colours[int(globalid)%32,:])
              number_of_matches= 1
              old_id = globalid
              estimated_position = [- newpoint[0]/newpoint[2], newpoint[1]/newpoint[2]]
            else: 
              number_of_matches +=1
              estimated_position[0]= estimated_position[0]*(number_of_matches-1)/number_of_matches - newpoint[0]/(newpoint[2]*number_of_matches)
              estimated_position[1]= estimated_position[1]*(number_of_matches-1)/number_of_matches + newpoint[1]/(newpoint[2]*number_of_matches)
        
        #plot the estimated projection point of first id 
        if (old_id!= -1 ):
          ax5.plot(estimated_position[0], estimated_position[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor=colours[int(old_id)%32,:])
          ax5.text(estimated_position[0], estimated_position[1], globalid,fontsize=14, color=colours[(int(globalid))%32,:])

      
        #plot the bounding boxes (+text),
        for seq, dets_tracker in Global_trackers.items():
      
          i = dic_axes[seq]

          frame = frame + timestamps[seq]

          #affiche image +partie 2  
          fn = os.path.join(path_to_train, seq, 'images', 'frame%d.jpg'%(frame-1))
          im =io.imread(fn)
          a,b,c = im.shape
          axes[i].imshow(im)
          axes[i].title.set_text(seq)

          frame = frame - timestamps[seq]

          for d in dets_tracker:
            if(display): #Afficher les trackers!!! 
              d = d.astype(np.int32) # frame, id, x1, y1, w, h 
              axes[i].text(d[2], d[3]-5, str(d[1]),fontsize=14, color=colours[(d[1])%32,:])
              axes[i].add_patch(patches.Rectangle((d[2],d[3]),d[4],d[5],fill=False,lw=3,ec=colours[d[1]%32,:]))
              #middle_point = (d[2]+ d[4]/2,d[3]+ d[5]/2)
              #newpoint = np.dot( np.linalg.inv(calibration_matrix),[(d[0]+ d[2])/2,(d[1]+ d[3])/2,1])

        ax5.title.set_text("Occupancy map")
        fig.canvas.flush_events()
        plt.draw()
        
        #enleve ancien bouding boxes
        for i,ax in enumerate( axes) :
              ax.cla()  



      all_trackers_old_time_frame = all_trackers_at_time_frame
    print("number feedback denied",feedback_count[0])
    print("number of feedback accepted",feedback_count[1])
    print("making last file")
    #file_output = 'data/trackers/S001-train/S001/data'
    number_of_frames =0
    patterns = os.path.join(path_to_train,'*', 'gt', 'gt.txt')

    with open(os.path.join(file_output,'all_cam.txt'),'w') as out_file:
      # for i,seq_dets_path_tracker in enumerate(glob.glob(pattern_output)):
      #   #print(seq_dets_path_tracker)
      #   seq = seq_dets_path_tracker[pattern_output.find('*'):].split(os.path.sep)[0] 

      for i,seq_dets_fn in enumerate(glob.glob(patterns)):

          seq = seq_dets_fn[patterns.find('*'):].split(os.path.sep)[0] 
          path = os.path.join(file_output,'%s.txt'%(seq))
          if seq != 'c005' and seq[0] =='c':
            all_seq_det_tracker = np.loadtxt(path, delimiter=',',ndmin=2)
            #print('all_seq_det_tracker',all_seq_det_tracker)s

            for d in all_seq_det_tracker:

                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(number_of_frames + d[0],d[1],d[2],d[3],d[4],d[5]),file=out_file)
            number_of_frames = number_of_frames + max_frame
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, min(framenumbers), min(framenumbers) / total_time)) 


#pas refait
def sort_cameras_online(args):
  print("sort online")
  outputfile = 'output_online'
  # all train
  display = args.display
  phase = args.phase #phase = train
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display



  axes = []
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(221, aspect='equal')
    ax2 = fig.add_subplot(222, aspect='equal')
    ax3 = fig.add_subplot(223, aspect='equal')
    ax4 = fig.add_subplot(224, aspect='equal')
    axes = [ax1,ax2,ax3,ax4,None]
  if not os.path.exists(outputfile):
    os.makedirs(outputfile)


  path_to_train = 'data/AIC21_Track3_MTMCTracking/train/S01'
  path_to_num_frames = 'data/AIC21_Track3_MTMCTracking/cam_framenum/S01.txt'

  framenumbers  =  np.loadtxt(path_to_num_frames,dtype=int,usecols = [1]) 
  print('framenumbers: ',framenumbers)
  print(max(framenumbers))
  print(min(framenumbers))

  pattern = os.path.join(path_to_train,'*', 'det', 'det_mask_rcnn.txt') # pattern :data//AIC21_Track3_MTMC_Tracking/train/s1/*/det/det_mask_rcnn.txt

  if not os.path.exists(path_to_train):
    sys.exit("ERROR: path to train does not exist")

  # BOUCLE pour chaque vidéo: create images from video
  number_of_cameras = len(glob.glob(pattern))
  print('number_of_cameras: ',number_of_cameras)
  for seq_dets_fn in glob.glob(pattern):
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 
    path_to_video = os.path.join(path_to_train,seq)
    print("transforming video to frame from camera ",seq)
    trans_video_to_frame(path_to_video)


  all_seq = [0]*number_of_cameras
  sort_trackers = [0]*number_of_cameras
  max_num_of_frames =0
  for i,seq_dets_fn in enumerate(glob.glob(pattern)):
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') 
    max_num_of_frames = max(max_num_of_frames, len(seq_dets))
    #print(len(seq_dets))
    sort_trackers[i] =  mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    all_seq[i]  = seq_dets

  for frame in range(min(framenumbers)):
      frame += 1 #detection and frame numbers begin at 1
      if frame%100==0:
          print("Processing frame %s."%(frame))
      for i,seq_dets_fn in enumerate(glob.glob(pattern)):
        seq_dets = all_seq[i]
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 
        with open(os.path.join(outputfile, '%s.txt'%(seq)),'w') as out_file:
          dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
          dets[:, 2:4] += dets[:, 0:2] #convert from [x1,y1,w,h] to [x1,y1,x2,y2] pour dans le update du tracker 
          total_frames += 1

          if(display and i != 4): #affiche image +partie 2
            fn = os.path.join(path_to_train, seq, 'images', 'frame%d.jpg'%(frame-1))
            im =io.imread(fn)
            axes[i].imshow(im)
            plt.title(seq + ' Tracked Targets')

          start_time = time.time()

          trackers = sort_trackers[i].update(dets) #IMPORTANT A CHEC

          cycle_time = time.time() - start_time
          total_time += cycle_time

          for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file) #ecrit dans le output file!! en mode [x1,y1,w,h] 
            if(display and i != 4): #Afficher les trackers!!! 
              d = d.astype(np.int32) # forme:[x1 y1 x2 y2   id] : [222 179 256 288   2]
              axes[i].add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:])) #ecrit en mode [x1,y1,w,h]


      if(display): #afficher imager 
        fig.canvas.flush_events()
        plt.draw()
        for i,ax in enumerate( axes) :
          if (i !=4): ax.cla() #supprimer ancien points et bbox dessiner

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")


def sort_monoc_dom_color(args):
  # IF display: tracking camera par camera
  # Sinon dominant color 
  # OUTPUT_COLOR

  print("sort offline color")
  # all train
  output_file = 'data/single_trackers/output_gt_with_color'
  display = args.display
  path_to_train = args.path_train
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display

  # plt.ion() # interactive mode
  # fig = plt.figure()

  if(display):
    plt.ion() # interactive mode
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists(output_file):
    os.makedirs(output_file)



  pattern = os.path.join(path_to_train,'*', 'gt', 'gt.txt') # pattern :data//AIC21_Track3_MTMC_Tracking/train/s1/*/det/det_mask_rcnn.txt
  #pattern = os.path.join(path_to_train,'*', 'det', 'det_mask_rcnn.txt') # pattern :data//AIC21_Track3_MTMC_Tracking/train/s1/*/det/det_mask_rcnn.txt


  if not os.path.exists(path_to_train):
    sys.exit("ERROR: path to train does not exist")



  # BOUCLE pour chaque vidéo: create images from video
  for seq_dets_fn in glob.glob(pattern):
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] 
    path_to_video = os.path.join(path_to_train,seq)
    print("transforming video to frame from camera ",seq)
    trans_video_to_frame(path_to_video)


  # BOUCLE pour chaque vidéo: create sort-tracker
  for seq_dets_fn in glob.glob(pattern):
    #print('seq_dets_fn: ',seq_dets_fn) # data/AIC21_Track3_MTMC_Tracking/train/S01/c002/det/det_mask_rcnn.txt ( juste l'endroit * qui va changer)
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker

    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  #list de list du fichier seq_dets_fn (det_mask_rcnn.txt)
 
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]  #just le nom de la sequence exemple ADL_Rundle ou c001

    with open(os.path.join(output_file, '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      #BOUCLE POUR CHAQUE FRAME
      for frame in range(int(seq_dets[:,0].max())):
        if frame%200==0:
          print("Processing frame %s."%(frame))

        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1
        fn = os.path.join(path_to_train, seq, 'images', 'frame%d.jpg'%(frame-1))
        im =io.imread(fn)
   
        if(display): #affiche image +partie 2
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()

        trackers = mot_tracker.update(dets) #IMPORTANT A CHEC

        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:

          im1 = im[int(d[1]):int(d[3])+1,int(d[0]):int(d[2])+1]
          #mean_color  = np.mean(im1, axis=(0, 1))
          dominant_color =get_dominant_color(im1)
          if dominant_color == [0,0,0]:
            dominant_color= np.mean(im1, axis=(0, 1))
          # print('mean_color:',mean_color)
          # print('dominant_color:',dominant_color)

          # plt.imshow(im1)
          # plt.show()
          # sleep(2)   
          # print(dominant_color)
          mean_color =dominant_color 

          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,%.2f,%.2f,%.2f'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1],mean_color[0],mean_color[1],mean_color[2]),file=out_file) #ecrit dans le output file!!
          if(display): #Afficher les trackers!!! 
            pass
            d = d.astype(np.int32) # forme:[x1 y1 x2 y2   id] : [222 179 256 288   2]
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        # Affiche le carré de detection original
        if False:
          for d2 in dets:
            if(display): #Afficher les trackers!!! 
              d2 = d2[0:4]
              d2 = d2.astype(np.int32)
              ax1.add_patch(patches.Rectangle((d2[0],d2[1]),d2[2]-d2[0],d2[3]-d2[1],fill=False,lw=3))

        if(display): #afficher imager 
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()
    

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")


if __name__ == '__main__':
  print('au dd')