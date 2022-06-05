import numpy as np

#indexes =  np.where(np.all(distances[:,[0,1]] == np.array(["A",2], dtype='object'), axis=1))
#indexes = np.append(indexes, np.where(np.all(distances[:,[0,1]] == np.array(["A",3], dtype='object'), axis=1)))
A=[3,2]
B = [1,1]

print(A-B)

exit()
distances = [
    #[0,0,0,0,10000],

    ["A",2, "B",4,'11'],
    ["A",2, "B",5,'12'],
    ["A",2, "C",6,'1.4'],
    ["A",2, "D",7,'11.0'],

    ["A",1, "B",4, '1.0'],
    ["A",1, "B",5, '4.0'],
    ["A",1, "C",6,'1.2'],
    ["A",1, "D",7,'5'],


    ["A",3, "B",4,'13'],
    ["A",3, "B",5,'2'],
    ["A",3, "C",6,'14'],
    ["A",3, "D",7,'2.4'],

    ["B",4, "C",6,'1.4'],
    ["B",4, "D",7,'1.5'],

    ["B",5, "C",6,'11.0'],
    ["B",5, "D",7,'2'],
    
    ["C",6, "D",7,'16']

    ]


distances = [
    #[0,0,0,0,10000],
    ["A",1, "B",2, '1.0'],
    ["A",1, "C",3, '1.1'],
    ["A",1, "D",4,'1.2'],
    ["A",1, "D",5,'2.1'],

    ["B",2, "C",3,'1.3'],
    ["B",2, "D",4,'1.4'],
    ["B",2, "D",5,'2.3'],
    
    ["C",3, "D",4,'1'],
    ["C",3, "D",5,'2']

    ]

distances = np.array(distances,dtype='object' )





global_count = 0

Global_trackers = {}

matches =np.array([['0',0,0]],dtype='object' )


#distances =  np.array([[0,0,0,0,10000]],dtype='object')



best_distance = 0.0

for i in range(0,10):
    if len(distances) != 0: 
        add = distances[:,4]
        add = add.astype(np.float64)
        index =np.argmin(add)
        seq,tracker_id,seq2,tracker2_id,distance = distances[index]
        # print( seq,tracker_id,seq2,tracker2_id,distance)
        # print("helooooooooo")
        # exit()
        distances = np.delete(distances,index, 0)
        if i ==0:
            best_distance = float(distance)

        if float(distance) > 4* best_distance:
            print("Now more good matches")
            break

        match1 = matches[ np.where(np.all(matches[:,[1,2]] == np.array([seq,str(tracker_id)], dtype='object'), axis=1))]
        match2 = matches[ np.where(np.all(matches[:,[1,2]] == np.array([seq2,str(tracker2_id)], dtype='object'), axis=1))]

        if len(match1) !=0 and len(match2) != 0:
            if (match1[0][0]) !=(match2[0][0]):
                indexesss = np.where(matches[:,[0]] == match2[0][0])
                for i in indexesss:
                    matches[i,0] = match1[0][0]
                print("ERROR ERROR ERROR, match between two different cars") #pas specialement un error mais cas tres particulier)  #normalement else et rien non? ca voudrait dire qu'ils sont tous les deux deja trouvé avec le meme global id

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


        #delete each item of seq, id1, seq2, ALL
        indexes = np.where(np.all(distances[:,[0,1,2]] == np.array([seq,tracker_id,seq2], dtype='object'), axis=1))
        distances = np.delete(distances,indexes,axis=0)
        indexes = np.where(np.all(distances[:,[0,2,3]] == np.array([seq2,seq,tracker_id], dtype='object'), axis=1))
        distances = np.delete(distances,indexes,axis=0)


        #recompute distances 
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


            
            # t2 = all_trackers_at_time_frame[s2][all_trackers_at_time_frame[s2][:,1]==float(id2)][0]
            # d2 = compute_distance(s2,t2,seq2,tracker2,all_cal_matrices)
            #other_distance = (number_of_cameras - 1)*d /number_of_cameras + d2/number_of_cameras

        indexes =  np.where(np.all(distances[:,[2,3]] == np.array([seq,tracker_id], dtype='object'), axis=1))
        #if(indexes[0].size != 0):
        for j in indexes[0]:
            s2,id2,s1,id1,d = distances[j,:]
            index1 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([s2,id2,seq2,tracker2_id], dtype='object'), axis=1))
            index2 = np.where(np.all(distances[:,[0,1,2,3]] == np.array([seq2,tracker2_id,s2,id2], dtype='object'), axis=1))
        
            newitem = distances[index1]
            if newitem.size != 0:
                newitem = newitem[0]
            else:
                newitem = distances[index2][0]
            distances[j,4] =str( float( d)/2 + float(newitem[4])/2)


        #delete each item seq2, id2
        indexes = np.where(np.all(distances[:,[0,1]] == np.array([seq2,tracker2_id], dtype='object'), axis=1))
        distances = np.delete(distances,indexes,axis=0)
        indexes = np.where(np.all(distances[:,[2,3]] == np.array([seq2,tracker2_id], dtype='object'), axis=1))
        distances = np.delete(distances,indexes,axis=0)
        
        # matches = np.append(matches,[[globalid, seq,tracker_id]],axis=0)
        # matches = np.append(matches,[[globalid,seq2,tracker2_id]], axis=0)

        if len(match1)==0:
            matches = np.append(matches,[[globalid, seq,tracker_id]],axis=0)
        if len(match2)==0:
            matches = np.append(matches,[[globalid,seq2,tracker2_id]], axis=0)


print(matches)





