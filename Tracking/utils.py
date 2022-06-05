from re import X
import numpy as np
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
np.seterr(divide='ignore', invalid='ignore')
from sklearn.cluster import KMeans
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from haversine import haversine, Unit


global_count = 0





def get_dominant_color(image, k=2):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
   
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    if image.shape != (0, 3):
        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        return list(dominant_color)
    else:
      return [0,0,0]


def compute_distance_rgb(trackerA,trackerB):
  #compute rgb distance beween two trackers
  #Form of tracker: frame, id, x, y, w,h,1, R,G,B
  lab1 = color.rgb2lab(trackerA[7:10])
  lab2 = color.rgb2lab(trackerB[7:10])
  color1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
  color2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
  delta_e = delta_e_cie1976(color1, color2)
  return delta_e

def on_image(x,y,w,h):

  if (x < 0) or (x+w>1920) or (y< 0) or (y+h>1080): 
    #print("specfial case denied")
    return False
  #print("special case accepted")
  return True


def compute_projetction_image_to_occup(matrice,point):
  gps = np.dot( np.linalg.inv(matrice),[point[0],point[1],1])
  gps = [i/gps[2] for i in gps]
  return gps[0:2]

def compute_projetction_point_occup_to_image(matrice,point):
  gps = np.dot( matrice,[point[0],point[1],1])
  gps = [i/gps[2] for i in gps]
  return gps[0:2]


def find_tracker(local_id,matches,all_trackers_at_time_frame):
    old_match_to_feedback =  matches[matches[:,2] == local_id]  #ajouter condition pour voir si on l'ajoute dans nouveau match
    glidA, seq, seq_id =  old_match_to_feedback[0]
    findtracker = all_trackers_at_time_frame[seq][all_trackers_at_time_frame[seq][:,1]==float(seq_id)]
    if findtracker.size != 0:
      tracker =findtracker[0]
      tracker = tracker.astype(np.int32)
      return tracker, seq
    else:
      return [],"not_found"

                        
def compute_distance(seqA, trackerA, seqB, trackerB, all_cal_matrices):
  #compute disntance of middlepoint of 2 bounding boxes, by projecting it with the homography matrix 
  #Input : seq     :  C001
  #        tracker  : frame, id, x1, y1, w, h ,1, R,G,B
  #         all_cal_matrices: dictonary with all homography matrices

  bbxA = trackerA[2:6]


  middlepointA = [bbxA[0]+bbxA[2]/2, bbxA[1]+bbxA[3]/2]
  gpsA = np.dot( np.linalg.inv(all_cal_matrices[seqA]),[middlepointA[0],middlepointA[1],1])
  gpsA = [i/gpsA[2] for i in gpsA]

  bbxB = trackerB[2:6]
  middlepointB = [bbxB[0]+bbxB[2]/2, bbxB[1]+bbxB[3]/2]
  gpsB = np.dot( np.linalg.inv(all_cal_matrices[seqB]),[middlepointB[0],middlepointB[1],1])
  gpsB = [i/gpsB[2] for i in gpsB]
  
  distance = ((gpsA[0] - gpsB[0])**2 + (gpsA[1] - gpsB[1])**2)**0.5

  #hs_distance = haversine(gpsA[0:2],gpsB[0:2],unit=Unit.METERS)
  return distance


def apply_clustering(args,all_trackers_at_time_frame,all_trackers_at_old_time_frame,all_cal_matrices,old_matches,global_count,feedback_count,feedback):
  '''
  Input: all trackers in dictronary per camera

  output: match of the cars that appears on minimum 2 cameras
      matches: [globalid, seq,tracker_id ]
      global_trackers: 
      feedback_trackers: [['3', 'c002', '8.0', 1190.9606740264114, 1209.1852866075683]]

  Description: 
      Etape 1: initialise all global variables (global_trackers,matches, distances)
      Etape 2: Compute all distances
      Etape 3: Match the bbx:
          A: find tracker
          B: delete all distance of first matched elem
          C: recompute all distances of second matched elem
          D: delete all distances from car in camera1 from car in camera2 

      Etape 4:: Matching phase + feedback


  '''

  '''ETAPE 1 '''
  #initialise Global_tracker, distances and matches
  feedback_trackers = []
  global_trackers = {}
  additional_tracker= {}
  for seq in all_trackers_at_time_frame.keys():
    global_trackers[seq]= []
    additional_tracker[seq]= []

 

  distances =  np.array([[0,0,0,0,'10000']],dtype='object')
  matches =np.array([['0',0,0]],dtype='object' )


  '''ETAPE 2'''
  #Compute all distances from each bbx to each other bbx not in the same image but at same timeframe
  for seq, trackers in all_trackers_at_time_frame.items():
    match_found = False
    for tracker in trackers:
        first_match = False
        nearest_tracker = []
        nearest_distance = 100000   
        best_seq = []
        for seq2, trackers2 in all_trackers_at_time_frame.items():
            nearest_distance = 100000  
            if seq2 != seq: 
                for tracker2 in trackers2:
                    distance_rgb = compute_distance_rgb(tracker,tracker2) / 4000000
                    distance = compute_distance(seq, tracker, seq2, tracker2, all_cal_matrices)
                    #distance = distance_rgb + distance
                    distances = np.append(distances,[[seq,tracker[1],seq2,tracker2[1],float(distance) ]], axis=0)
  


  ''' ETAPE 3'''
  #best_distance = 0.0
  best_distance = 0.00012706700631763762
  file_name =  args.BENCHMARK
  if file_name == "S01": bd = 0.00016
  else: bd = 0.0001


  #Match the bbx based on the distances
  for i in range(0,20):
    add = distances[:,4]
    add = add.astype(np.float64)
    index =np.argmin(add)


    seq,tracker_id,seq2,tracker2_id,distance = distances[index]

    if float(distance) > bd:
 
    #if float(distance) > 0.0001:
        #print("No more good matches")
        break
    #look if allready match one of the bbx with another one  
    match1 = matches[ np.where(np.all(matches[:,[1,2]] == np.array([seq,str(tracker_id)], dtype='object'), axis=1))]
    match2 = matches[ np.where(np.all(matches[:,[1,2]] == np.array([seq2,str(tracker2_id)], dtype='object'), axis=1))]


    #Case if We have two bbx with id 1 and two bbx with id2, and a match between the for, to take same id for each of them
    if len(match1) !=0 and len(match2) != 0:
      if (match1[0][0]) !=(match2[0][0]):
          if (match1[0][0]) !=(match2[0][0]):
              indexesss = np.where(matches[:,[0]] == match2[0][0])
              for i in indexesss:
                  matches[i,0] = match1[0][0]

    number_of_cameras = 2

    if len(match1)!=0:
      globalid = int(match1[0][0])
      number_of_cameras = len(matches[ np.where(matches[:,[0]] == globalid)]) + 1
    elif len(match2)!=0:
      globalid = int(match2[0][0])
      number_of_cameras = len(matches[ np.where(matches[:,[0]] == globalid)]) + 1
    else:
      globalid = global_count
      global_count += 1

    #find tracker to put it global tracker
    findtracker = all_trackers_at_time_frame[seq][all_trackers_at_time_frame[seq][:,1]==float(tracker_id)]
    tracker =findtracker[0]
    if tracker ==[]:
      print(findtracker)
    tracker2 =all_trackers_at_time_frame[seq2][all_trackers_at_time_frame[seq2][:,1]==float(tracker2_id)][0]


    #delete each item of seq, id1, seq2, ALL
    indexes = np.where(np.all(distances[:,[0,1,2]] == np.array([seq,tracker_id,seq2], dtype='object'), axis=1))
    distances = np.delete(distances,indexes,axis=0)
    indexes = np.where(np.all(distances[:,[0,2,3]] == np.array([seq2,seq,tracker_id], dtype='object'), axis=1))
    distances = np.delete(distances,indexes,axis=0)


    #recompute distances 
    newmethod = True

    if newmethod:
        indexes = np.where(np.all(distances[:,[0,1]] == np.array([seq,tracker_id], dtype='object'), axis=1))
        for j in indexes[0]:
            elem = distances[j,:]
            [s1,id1,s2,id2,d] = distances[j,:]
            index1 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([s2,id2,seq2,tracker2_id], dtype='object'), axis=1))
            index2 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([seq2,tracker2_id,s2,id2], dtype='object'), axis=1))
            newitem = distances[index1]
            if newitem.size != 0:
                newitem = newitem[0]
            elif distances[index2].size != 0:
                newitem = distances[index2][0]
            else: 
                newitem =  [0,0,0,0,'10000']
            distances[j,4] =str( float( d)/2 + float(newitem[4])/2) 



        indexes =  np.where(np.all(distances[:,[2,3]] == np.array([seq,tracker_id], dtype='object'), axis=1))
        #if(indexes[0].size != 0):
        for j in indexes[0]:
            s2,id2,s1,id1,d = distances[j,:]
            index1 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([s2,id2,seq2,tracker2_id], dtype='object'), axis=1))
            index2 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([seq2,tracker2_id,s2,id2], dtype='object'), axis=1))
        
            newitem = distances[index1]
            if newitem.size != 0:
                newitem = newitem[0]
            elif distances[index2].size != 0:
                newitem = distances[index2][0]
            else: 
                newitem =  [0,0,0,0,'10000']
            distances[j,4] =str( float( d)/2 + float(newitem[4])/2) 


    else:
      indexes = np.where(np.all(distances[:,[0,1]] == np.array([seq,tracker_id], dtype='object'), axis=1))
      for j in indexes[0]:
        s1,id1,s2,id2,d = distances[j,:]
        t2 = all_trackers_at_time_frame[s2][all_trackers_at_time_frame[s2][:,1]==float(id2)][0]
        d2 = compute_distance(s2,t2,seq2,tracker2,all_cal_matrices)
        distances[j,4] = str(float( d)/2 + float(d2)/2)

 
    #indexes = np.append(indexes, np.where(np.all(distances[:,[2,3]] == np.array([seq,tracker_id], dtype='object'), axis=1)))
    indexes =  np.where(np.all(distances[:,[2,3]] == np.array([seq,tracker_id], dtype='object'), axis=1))
    for j in indexes[0]:
      s2,id2,s1,id1,d = distances[j,:]
      t2 = all_trackers_at_time_frame[s2][all_trackers_at_time_frame[s2][:,1]==float(id2)][0]
      d2 = compute_distance(s2,t2,seq2,tracker2,all_cal_matrices)
      distances[j,4] = str(float( d)/2 + float(d2)/2)


    #delete each item seq2, id2
    indexes = np.where(np.all(distances[:,[0,1]] == np.array([seq2,tracker2_id], dtype='object'), axis=1))
    distances = np.delete(distances,indexes,axis=0)
    indexes = np.where(np.all(distances[:,[2,3]] == np.array([seq2,tracker2_id], dtype='object'), axis=1))
    distances = np.delete(distances,indexes,axis=0)

    # matches = np.append(matches,[[globalid, seq,tracker[1]]],axis=0)
    # matches = np.append(matches,[[globalid,seq2,tracker2[1]]], axis=0)


    if len(match1)==0:
        #global_trackers[seq].append(np.concatenate(([tracker[0],globalid],tracker[2:])))
        matches = np.append(matches,[[globalid, seq,tracker[1]]],axis=0)
    if len(match2)==0:
        #global_trackers[seq2].append(np.concatenate(([tracker2[0],globalid],tracker2[2:])))
        matches = np.append(matches,[[globalid,seq2,tracker2[1]]], axis=0)



  ''' ETAPE 4'''
  #boucle sur tous les nouveau global id:
  # dedans on vait une boucle sur tous les locals ids:

  matches = np.delete(matches, (0), axis=0) #delete first row = 0000
  matches = matches[matches[:,0].argsort()]

  (ids, list_indexes) =  np.unique(matches[:,0], return_index=True, return_inverse=False, return_counts=False, axis=None)
  number_ids = len(list_indexes)
  number_of_matches = len(matches)

  feedback_found = {}
  
  #boucles sur new global ids 
  for i,id in enumerate(ids):
    first_index = list_indexes[i]
    if i+1 ==number_ids: 
      last_index =number_of_matches
    else:
      last_index = list_indexes[i+1]

    #trouver tous les locals ids du new same global id
    all_new_ids = matches[first_index:last_index,2]
    all_new_seq = matches[first_index:last_index,1]

    #boucle sur tous les locals ids, check si trouve dans old matches
    for local_id  in all_new_ids:
      old_combi =  old_matches[old_matches[:,2] == local_id,:]
      if old_combi.size != 0:
        old_global_id =old_combi[0,0]

        old_ids =  old_matches[old_matches[:,0] == old_global_id,2]
        inters_ids = np.intersect1d(all_new_ids, old_ids)


        if len(inters_ids)>=1: #combien avoir en commun avec old_id pour match meme global id maintenant
          #cas special break pas le meme: 4 caisse avant 2 un global id, 2 un autre  global id ( faudrait prendre celui avec plus petite distance)
          if old_global_id not in matches[:,0]:
            matches[first_index:last_index,0] = old_global_id # pourrait delete de old matches

            '''FEEDBACK'''
            #FEEDBACK: TO OPTIMIZE SINGLE TRACKERS
            if(feedback):
              number_new_bbx,number_old_bbx,number_of_inter = len(all_new_ids),len(old_ids),len(inters_ids)
              #changer condition exemple: C1,2,3 puis C1,2,4 => restaurer 3

              #if  number_new_bbx < number_old_bbx:
              #  if  number_new_bbx == number_of_inter:

              if  number_of_inter < number_old_bbx:

                    ids_to_add = np.setxor1d(all_new_ids, old_ids) # te donne tous les elem que dans une des deux liste mais du coup aussi les nouveau...

                    #if len(ids_to_add) ==1:
                    for id_to_add in ids_to_add:
                      if id_to_add in old_ids:
                        if len(inters_ids) != 0:
                          trackerA,seqA = find_tracker(id_to_add,old_matches,all_trackers_at_old_time_frame) 

                          if seqA not in all_new_seq:#check pas match autre detection sur camera

                            #etape 1 trouver l'ancien tracker: Tracker A
                            newpointA = [ trackerA[2]+trackerA[4]/2,trackerA[3]+trackerA[5]/2 ]
                            newpointA = compute_projetction_image_to_occup(all_cal_matrices[seqA],newpointA)

                            #etape 2 trouver autre ancien point : tracker B
                            trackerB,seqB = find_tracker(inters_ids[0],old_matches,all_trackers_at_old_time_frame) 
                            newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
                            newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)


                            #etape 3 calculer la distance
                            distance= [newpointA[0] - newpointB[0], newpointA[1] - newpointB[1]]
          

                            #etape 4: calculer nouveau point de tracker B
                            trackerB,seqB = find_tracker(inters_ids[0],matches,all_trackers_at_time_frame) 
                            newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
                            newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)

                            #etape 5 calculer le nouveau point du tracker A
                            estimated_pos1 = [newpointB[0]+distance[0] ,newpointB[1]+ distance[1]]
                            estimated_pos = compute_projetction_point_occup_to_image(all_cal_matrices[seqA],estimated_pos1)
                            
                            if on_image(estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2,trackerA[4],trackerA[5]):
                              feedback_count[0]+=1
                              feedback_trackers.append([id,seqA,id_to_add,estimated_pos[0],estimated_pos[1],estimated_pos1[0],estimated_pos1[1]])
                              trackerA[0] = trackerA[0]+1 #on new frame
                              trackerA[2],trackerA[3] = estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2


                              all_trackers_at_time_frame[seqA]= np.append(all_trackers_at_time_frame[seqA], [trackerA],axis=0)
                              matches = np.append(matches,[[old_global_id, seqA,float(trackerA[1])]],axis=0)

                            else: feedback_count[1]+=1
                            


            break

          
        


  


  '''FEEDBACK'''
  if(args.detections != "yolo" and feedback):
    #FEEDBACK: TO OPTIMIZE SINGLE TRACKERS
    old_matches = old_matches[old_matches[:,0].argsort()]
    (old_ids, list_indexes) =  np.unique(old_matches[:,0], return_index=True, return_inverse=False, return_counts=False, axis=None)
    (new_ids, list_indexes_new) =  np.unique(matches[:,0], return_index=True, return_inverse=False, return_counts=False, axis=None)
    number_old_ids = len(list_indexes)
    number_of_old_matches = len(old_matches)
  


    for i,old_id in enumerate(old_ids):
      if old_id in new_ids:
        pass
      else:
        first_index = list_indexes[i]
        if i+1 ==number_old_ids: 
          last_index =number_of_old_matches
        else:
          last_index = list_indexes[i+1]

        #trouver tous les locals ids du new same global id
        all_old_ids = old_matches[first_index:last_index,2]
        all_old_seq = old_matches[first_index:last_index,1]

        no_more = True
        #boucle sur tous les locals ids, check si trouve dans matches
        for old_local_id  in all_old_ids:
          old_combi =  matches[matches[:,2] == old_local_id,:]
          if old_combi.size != 0:
            no_more = False
            #print("wtf case")
            break
        if no_more:
          if len(all_old_ids) ==2:


            counter_found =0
            for old_local_id  in all_old_ids:
              tracker,seq = find_tracker(old_local_id,old_matches,all_trackers_at_time_frame) 
              if seq != "not_found":
                counter_found +=1
                trackerB_new,seqB_new,final_id= tracker,seq,old_local_id

            if counter_found ==1:
              for old_local_id  in all_old_ids:
                if old_local_id != final_id:
                  # tracker A celui compute new point: => old_local_id
                  # tracker B: celui a le id 

                  #etape 1 trouver l'ancien tracker: Tracker A
                  trackerA,seqA = find_tracker(old_local_id,old_matches,all_trackers_at_old_time_frame) 
                  newpointA = [ trackerA[2]+trackerA[4]/2,trackerA[3]+trackerA[5]/2 ]
                  newpointA = compute_projetction_image_to_occup(all_cal_matrices[seqA],newpointA)

                  #etape 2 trouver autre ancien point : tracker B
                  trackerB,seqB = find_tracker(final_id,old_matches,all_trackers_at_old_time_frame) 
                  newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
                  newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)


                  #etape 3 calculer la distance
                  distance= [newpointA[0] - newpointB[0], newpointA[1] - newpointB[1]]


                  #etape 4: calculer nouveau point de tracker B
                  trackerB,seqB = trackerB_new,seqB_new
                  newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
                  newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)

                  #etape 5 calculer le nouveau point du tracker A
                  estimated_pos1 = [newpointB[0]+distance[0] ,newpointB[1]+ distance[1]]
                  estimated_pos = compute_projetction_point_occup_to_image(all_cal_matrices[seqA],estimated_pos1)
                  
                  if on_image(estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2,trackerA[4],trackerA[5]):
                    feedback_count[0]+=1
                    feedback_trackers.append([old_id,seqA,old_local_id,estimated_pos[0],estimated_pos[1],estimated_pos1[0],estimated_pos1[1]])
                    trackerA[0] = trackerA[0]+1 #on new frame
                    trackerA[2],trackerA[3] = estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2


                    all_trackers_at_time_frame[seqA]= np.append(all_trackers_at_time_frame[seqA], [trackerA],axis=0)
                    matches = np.append(matches,[[old_id, seqA,float(trackerA[1])]],axis=0)
                    matches = np.append(matches,[[old_id, seqB,float(trackerB[1])]],axis=0)

                  else: feedback_count[1]+=1

            



  #       else:
  #         print("other special case")



  #         old_ids =  old_matches[old_matches[:,0] == old_global_id,2]
  #         inters_ids = np.intersect1d(all_new_ids, old_ids)


  #         if len(inters_ids)>=2: #combien avoir en commun avec old_id pour match meme global id maintenant
  #           #cas special break pas le meme: 4 caisse avant 2 un global id, 2 un autre  global id ( faudrait prendre celui avec plus petite distance)
  #           if old_global_id not in matches[:,0]:
  #             matches[first_index:last_index,0] = old_global_id # pourrait delete de old matches


            # if (feedback):
            #   number_new_bbx,number_old_bbx,number_of_inter = len(all_new_ids),len(old_ids),len(inters_ids)
            #   #changer condition exemple: C1,2,3 puis C1,2,4 => restaurer 3

            #   #if  number_new_bbx < number_old_bbx:
            #   #  if  number_new_bbx == number_of_inter:
            #   if  number_of_inter < number_old_bbx:

            #         ids_to_add = np.setxor1d(all_new_ids, old_ids) #pas 100 correct te donne tous les elem que dans une des deux liste mais du coup aussi les nouveau...

            #         #if len(ids_to_add) ==1:
            #         for id_to_add in ids_to_add:
            #           if id_to_add in old_ids:


            #             if len(inters_ids) != 0:
            #               trackerA,seqA = find_tracker(id_to_add,old_matches,all_trackers_at_old_time_frame) 

            #               if seqA not in all_new_seq:
            #                 #etape 1 trouver l'ancien tracker: Tracker A
            #                 newpointA = [ trackerA[2]+trackerA[4]/2,trackerA[3]+trackerA[5]/2 ]
            #                 newpointA = compute_projetction_image_to_occup(all_cal_matrices[seqA],newpointA)

            #                 #etape 2 trouver autre ancien point : tracker B
            #                 trackerB,seqB = find_tracker(inters_ids[0],old_matches,all_trackers_at_old_time_frame) 
            #                 newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
            #                 newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)


            #                 #etape 3 calculer la distance
            #                 distance= [newpointA[0] - newpointB[0], newpointA[1] - newpointB[1]]
          

            #                 #etape 4: calculer nouveau point de tracker B
            #                 trackerB,seqB = find_tracker(inters_ids[0],matches,all_trackers_at_time_frame) 
            #                 newpointB = [ trackerB[2]+trackerB[4]/2,trackerB[3]+trackerB[5]/2 ]
            #                 newpointB = compute_projetction_image_to_occup(all_cal_matrices[seqB],newpointB)

            #                 #etape 5 calculer le nouveau point du tracker A
            #                 estimated_pos1 = [newpointB[0]+distance[0] ,newpointB[1]+ distance[1]]
            #                 estimated_pos = compute_projetction_point_occup_to_image(all_cal_matrices[seqA],estimated_pos1)
                            
            #                 if on_image(estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2,trackerA[4],trackerA[5]):
            #                   feedback_count[0]+=1
            #                   feedback_trackers.append([local_id,seqA,id_to_add,estimated_pos[0],estimated_pos[1],estimated_pos1[0],estimated_pos1[1]])
            #                   trackerA[0] = trackerA[0]+1 #on new frame
            #                   trackerA[2],trackerA[3] = estimated_pos[0]-trackerA[4]/2,estimated_pos[1]-trackerA[5]/2


            #                   all_trackers_at_time_frame[seqA]= np.append(all_trackers_at_time_frame[seqA], [trackerA],axis=0)
            #                   matches = np.append(matches,[[old_global_id, seqA,trackerA[1]]],axis=0)

            #                 else: feedback_count[1]+=1
                              
 

            #   break
        


          

  for global_id, seq, seq_id in matches:
      findtracker = all_trackers_at_time_frame[seq][all_trackers_at_time_frame[seq][:,1]==float(seq_id)]
      tracker =findtracker[0]
      tracker = tracker.astype(np.int32)
      global_trackers[seq].append(np.concatenate(([tracker[0],float(global_id)],tracker[2:])))


  return global_trackers, matches, feedback_trackers, feedback_count


