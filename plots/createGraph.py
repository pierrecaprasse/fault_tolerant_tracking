# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:28:17 2017

@author: Diederik Coppitters
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
code example:
createGraph(X,Y,['-']*len(X), ['legend']*len(X), X[0],Y[0], r'$n$','rpm',r'$\eta_\mathrm{el}$','', 0.9, -0.25, -0.2, 0.7, [X[0],X[-1],Y[0],Y[-1]],'result','pdf', addTick=True,extraTick=[X[3],Y[3]]):
'''  

def createGraph(x,y,PLotType, legend, legx,legy, xSymbol,xUnit,ySymbol,yUnit, xx, yx, xy, yy, limits,filename, extension, font='serif', size_ticks=14, size_label=18, color = 'black',addTickXY=1,extraTickXY=[1,1],addTickX=1,extraTickX=[1,1],addTickY=1,extraTickY=[1,1],dataNotVisibleOutLimits=False, extraColor=False, ColorNames = []):
  
  def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 35))  # outward by 10 points
           # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
      
  length = len(x)  
  p = []
  fontdic = []
  plt.rcParams['mathtext.fontset'] = 'stix'
  if extraColor == False:
      for i in range(length):
          p = plt.plot(x[i],y[i],PLotType[i], clip_on = dataNotVisibleOutLimits)
          fontdic.append({'family': font,
            'color':  p[0].get_color(),
            'weight': 'normal',
            'size': size_label,
            })
  if extraColor == True:
      for i in range(length):
          p = plt.plot(x[i],y[i],PLotType[i], color=ColorNames[i][:] , clip_on = dataNotVisibleOutLimits)
          fontdic.append({'family': font,
            'color':  p[0].get_color(),
            'weight': 'normal',
            'size': size_label,
            })

    

  plt.axis(limits)

  if addTickX == 0:
    plt.xticks([limits[0],limits[1]])
    #plt.yticks([limits[2],limits[3]])
  elif addTickX == 1:
    plt.xticks([limits[0],extraTickX[0],limits[1]])
    #plt.yticks([limits[2],extraTickX[1],limits[3]])
  elif addTickX == 2:
    plt.xticks([limits[0],extraTickX[0],extraTickX[1],limits[1]])
    #plt.yticks([limits[2],extraTickX[1],extraTickX[3],limits[3]])
  elif addTickX == 3:
    plt.xticks([limits[0],extraTickX[0],extraTickX[1],extraTickX[2],limits[1]])
    #plt.yticks([limits[2],extraTickX[1],extraTickX[3],extraTickX[5],limits[3]])
  elif addTickX == 4:
    plt.xticks([limits[0],extraTickX[0],extraTickX[1],extraTickX[2],extraTickX[3],limits[1]])
    #plt.yticks([limits[2],extraTickX[1],extraTickX[3],extraTickX[5],extraTickX[7],limits[3]])
  elif addTickX == 5:
    plt.xticks([limits[0],extraTickX[0],extraTickX[1],extraTickX[2],extraTickX[3],extraTickX[4],limits[1]])
  elif addTickX == 6:
    plt.xticks([limits[0],extraTickX[0],extraTickX[1],extraTickX[2],extraTickX[3],extraTickX[4],extraTickX[5],limits[1]])
    #plt.yticks([limits[2],extraTickX[1],extraTickX[3],extraTickX[5],extraTickX[7],extraTickX[9],limits[3]])
  if addTickY == 0:
    #plt.xticks([limits[0],limits[1]])
    plt.yticks([limits[2],limits[3]])
  elif addTickY == 1:
    #plt.xticks([limits[0],extraTickY[0],limits[1]])
    plt.yticks([limits[2],extraTickY[0],limits[3]])
  elif addTickY == 2:
    #plt.xticks([limits[0],extraTickY[0],extraTickY[1],limits[1]])
    plt.yticks([limits[2],extraTickY[0],extraTickY[1],limits[3]])
  elif addTickY == 3:
    #plt.xticks([limits[0],extraTickY[0],extraTickY[1],extraTickY[2],limits[1]])
    plt.yticks([limits[2],extraTickY[0],extraTickY[1],extraTickY[2],limits[3]])
  elif addTickY == 4:
    #plt.xticks([limits[0],extraTickY[0],extraTickY[1],extraTickY[2],extraTickY[3],limits[1]])
    plt.yticks([limits[2],extraTickY[0],extraTickY[1],extraTickY[2],extraTickY[3],limits[3]])
  elif addTickY == 5:
    #plt.xticks([limits[0],extraTickY[0],extraTickY[1],extraTickY[2],extraTickY[3],extraTickY[4],limits[1]])
    plt.yticks([limits[2],extraTickY[0],extraTickY[1],extraTickY[2],extraTickY[3],extraTickY[4],limits[3]])
  
  ax = plt.subplot(111)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  #minor_ticksx = np.arange(1000, 1201, 20)                                               
  #minor_ticksy = np.arange(550, 651, 10)
  #ax.set_xticks(minor_ticksx, minor=True)                                           
  #ax.set_yticks(minor_ticksy, minor=True)                                                                                               

  #ax.grid(which='both')                                                            

  ax.yaxis.set_label_coords(xy,yy)
  ax.xaxis.set_label_coords(xx,yx)
  adjust_spines(ax, ['left', 'bottom'])
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')
  ax.tick_params(axis='both', direction = 'in', colors=color)
  ax.spines['left'].set_color(color)
  ax.spines['bottom'].set_color(color)
  for i in range(length):
    plt.text(legx[i], legy[i], legend[i], fontdict=fontdic[i] )
  for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(size_ticks)
    item.set_fontname(font)
    
  for item in ([ ax.xaxis.label, ax.yaxis.label] ):
    item.set_fontsize(size_label)
    item.set_fontname(font)
  
  plt.savefig(filename + '.' + extension, bbox_inches='tight')


def create_own_graph2(X,Y,labels,title,leg_loc,legx,legy,filename,xUnit='',yUnit=''):
   
  n = len(X)
  for i in range(n):
    # plotting the line 1 points
    plt.plot(X[i], Y[i], label = labels[i])


  # naming the x axis
  if xUnit == '':
      plt.xlabel(legx)
  else:
      plt.xlabel(legx + ' [' + xUnit + ']')

  # naming the y axis
  if yUnit == '':
      #plt.ylabel(legy).set_rotation(0)
      plt.ylabel(legy)
  else:
      plt.ylabel(legy + '\n' + '[' + yUnit + ']')

  # giving a title to my graph
  plt.title(title)

  # show a legend on the plot
  plt.legend( loc =leg_loc)

  # function to show the plot
  extension = 'pdf'
  plt.savefig(filename + '.' + extension, bbox_inches='tight')
  plt.show()


def create_own_graph(x1,y1,x2,y2,label1,label2,title,legx,legy,filename,xUnit='',yUnit=''):
   
   
    # plotting the line 1 points
    plt.plot(x1, y1, label = label1)

    # plotting the line 2 points
    plt.plot(x2, y2, label = label2)

    # naming the x axis
    if xUnit == '':
        plt.xlabel(legx)
    else:
        plt.xlabel(legx + ' [' + xUnit + ']')

    # naming the y axis
    if yUnit == '':
        #plt.ylabel(legy).set_rotation(0)
        plt.ylabel(legy)
    else:
        plt.ylabel(legy + '\n' + '[' + yUnit + ']')

    # giving a title to my graph
    plt.title(title)

    # show a legend on the plot
    plt.legend( loc ="upper right")

    # function to show the plot
    extension = 'pdf'
    plt.savefig(filename + '.' + extension, bbox_inches='tight')
    plt.show()





if __name__ == '__main__':
  # line 2 points
  case = 1
  cam_defected = 4
  cam_plot = 0# 0 => all
  file_name = "S01"

  if case ==1:
    leg_loc = "upper right"
    if cam_plot ==0:
      cam_to_plot = "all_cam"
      title = "Evolution of the global HOTA score" 
    elif cam_plot<11: 
      cam_to_plot = 'c00'+str( cam_plot )
      title = "Evolution of HOTA score of cam " +str(cam_plot) + ", from " + file_name
    else:
      print("error value cam_to_plot")
      exit()



    #inputfolder = 'data/HOTA_results/' + file_name + '/' +  'defected_cam_%s'%(cam_defected)
    #outputfolder = 'data/HOTA_results/' + file_name + '/' +'plots' + '/defected_cam_%s'%(cam_defected)
    inputfolder = 'data/HOTA_results/' + file_name + '/yolo/' +  'defected_cam_%s'%(cam_defected)
    outputfolder = 'data/HOTA_results/' + file_name + '/yolo/' +'plots' + '/defected_cam_%s'%(cam_defected)

    if not os.path.exists(outputfolder):
      os.makedirs(outputfolder)

    inputfile_feedback = inputfolder +'/feedback_save.txt'
    inputfile_no_feedback = inputfolder +'/no_feedback_save.txt'

    # inputfile_feedback = inputfolder +'/feedback.txt'
    # inputfile_no_feedback = inputfolder +'/no_feedback.txt'

    # data_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 
    # data_no_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 


    data_feedback = pd.read_csv( inputfile_feedback, sep=',')
    data_no_feedback = pd.read_csv( inputfile_no_feedback, sep=',')
    #print(data_feedback)
    seqs = data_feedback.head(0)
    perc = data_feedback['percentage']

    x1 = perc
    y1 = data_feedback[cam_to_plot]
    label1 = "with feedback"

    x2 = perc
    y2 = data_no_feedback[cam_to_plot]
    label2 = "no feedback"

    if cam_defected == 0:
      xlabel = 'missing detections of all cameras '
    else:
      xlabel = 'missing detections of cam '+ str(cam_defected)
    xunit = "%"
    ylabel = 'HOTA Score'
    yunit = ""

    filename = outputfolder+'/def_cam_' + str(cam_defected) + '_plotted_cam_'+str(cam_to_plot)
    #create_own_graph(x1,y1,x2,y2,label1,label2,title,xlabel,ylabel,filename,xunit,yunit)
    create_own_graph2([x1,x2],[y1,y2],[label1,label2],title,leg_loc, xlabel,ylabel,filename,xunit,yunit)


  else:
    title = "Evolution of HOTA score" 
    leg_loc = "lower left"
    inputfolder = 'data/HOTA_results/' + file_name + '/' +  'defected_cam_%s'%(cam_defected)
    outputfolder = 'data/HOTA_results/' + file_name + '/' +'plots' + '/defected_cam_%s'%(cam_defected)

    if not os.path.exists(outputfolder):
      os.makedirs(outputfolder)

    inputfile_feedback = inputfolder +'/feedback_save.txt'
    inputfile_no_feedback = inputfolder +'/no_feedback_save.txt'
    # inputfile_feedback = inputfolder +'/feedback.txt'
    # inputfile_no_feedback = inputfolder +'/no_feedback.txt'

    # data_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 
    # data_no_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 


    data_feedback = pd.read_csv( inputfile_feedback, sep=',')
    data_no_feedback = pd.read_csv( inputfile_no_feedback, sep=',')
    #print(data_feedback)
    seqs = data_feedback.head(0)
    perc = data_feedback['percentage']
    
    X = []
    Y = []
    labels = []

    for seq in seqs:
      if seq != 'percentage':
        X.append(perc)
        Y.append(data_feedback[seq])
        if seq =="all_cam":
          labels.append("Global")
        else:
          labels.append("Cam "+seq[-1])
 

    if cam_defected == 0:
      xlabel = 'missing detections of all cameras '
    else:
      xlabel = 'missing detections of cam '+ str(cam_defected)
    xunit = "%"
    ylabel = 'HOTA Score'
    yunit = ""

    filename = outputfolder+'/def_cam_' + str(cam_defected) + 'cams'
    #create_own_graph(x1,y1,x2,y2,label1,label2,title,xlabel,ylabel,filename,xunit,yunit)
    create_own_graph2(X,Y,labels,title,leg_loc,xlabel,ylabel,filename,xunit,yunit)







if __name__ == '__main__2':
  # line 2 points
  cam_defected = 4
  cam_plot = 3 # 0 => all
  file_name = "S01"


  if cam_plot ==0:
    cam_to_plot = "all_cam"
    title = "Evolution of the general HOTA score" 
  elif cam_plot<5: 
    cam_to_plot = 'c00'+str( cam_plot )
    title = "Evolution of HOTA score of cam " +str(cam_plot) + ", from " + file_name
  else:
    print("error value cam_to_plot")
    exit()



  inputfolder = 'data/HOTA_results/' + file_name + '/' +  'defected_cam_%s'%(cam_defected)
  outputfolder = 'data/HOTA_results/' + file_name + '/' +'plots' + '/defected_cam_%s'%(cam_defected)

  if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

  inputfile_feedback = inputfolder +'/feedback_save.txt'
  inputfile_no_feedback = inputfolder +'/no_feedback_save.txt'

  # inputfile_feedback = inputfolder +'/feedback.txt'
  # inputfile_no_feedback = inputfolder +'/no_feedback.txt'

  # data_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 
  # data_no_feedback = np.loadtxt(inputfile_feedback, delimiter=',') 


  data_feedback = pd.read_csv( inputfile_feedback, sep=',')
  data_no_feedback = pd.read_csv( inputfile_no_feedback, sep=',')
  #print(data_feedback)
  seqs = data_feedback.head(0)
  perc = data_feedback['percentage']

  x1 = perc
  y1 = data_feedback[cam_to_plot]
  label1 = "with feedback"

  x2 = perc
  y2 = data_no_feedback[cam_to_plot]
  label2 = "no feedback"

  xlabel = 'missing detections of cam '+ str(cam_defected)
  xunit = "%"
  ylabel = 'HOTA Score'
  yunit = ""

  filename = outputfolder+'/def_cam_' + str(cam_defected) + '_plotted_cam_'+str(cam_to_plot)
  #create_own_graph(x1,y1,x2,y2,label1,label2,title,xlabel,ylabel,filename,xunit,yunit)
  create_own_graph2([x1,x2],[y1,y2],[label1,label2],title,xlabel,ylabel,filename,xunit,yunit)



  sort1c_nf = []
  sortallcam_nf = []
  clustering = []