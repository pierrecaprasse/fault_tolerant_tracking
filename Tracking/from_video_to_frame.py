from sre_constants import OP_UNICODE_IGNORE
import cv2
import os

def trans_video_to_frame(path_to_folder_with_video):

    path_to_v = os.path.join(path_to_folder_with_video, 'vdo.avi')
    vidcap = cv2.VideoCapture(path_to_v)
    success,image = vidcap.read()
    count = 0
    if not os.path.exists(path_to_folder_with_video+'/images'):
      os.makedirs(path_to_folder_with_video+'/images')
      while success:
  
          output_path = os.path.join(path_to_folder_with_video, "images/frame%d.jpg" % count) 
          cv2.imwrite(output_path,image)
          success,image = vidcap.read()
          count += 1
          if count%100==0:
              print(count)
      print("total count:",count)

    # else:
    #   print("images already accessible")
  

   

