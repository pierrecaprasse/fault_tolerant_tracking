import os,sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import operator
import plotly
import orca
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
#import plotly.plotly as py
import chart_studio.plotly as chart
if not os.path.exists("images"):
    os.mkdir("/home/User/Desktop/newplot (1).png")


path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(path,'POST_PROCESS'))


import post_process_lib as lb
#matplotlib.rc('xtick', labelsize=7) 
#matplotlib.rc('ytick', labelsize=7) 

tc = 'PB_BAT_HP_TEST_STRAT5' # 0me case
my_post_process = lb.post_process(tc) 


eval_type = 'ROB' # l15-l26 pareto front plot,comment if not necessary  
result_dir = ['TEST']

LIGHT = False

#my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)

## L24 à 73 c'est CDF
    
#fig, ax = plt.subplots(3,1,gridspec_kw={'width_ratios': [1,1, 1]})
# fig, ax = plt.subplots(3,1,gridspec_kw={'width_ratios': [ 2,5,1]})
# Strat2=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/UQ/opt_design/data_cdf_lcox")
# datastrat2=np.loadtxt(Strat2,skiprows=1)
# 
# #plt.legend('reference')
# 
# Strat3=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/UQ/opt_design/data_cdf_lcox")
# datastrat3=np.loadtxt(Strat3,skiprows=1)
# 
# 
# ax[1].plot(datastrat3[:,0],datastrat3[:,1],color='lightgray',linewidth=0.5)
# ax[2].plot(datastrat3[:,0],datastrat3[:,1],color='lightgray',linewidth=0.5)
# 
# 
# 
# Strat4=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/UQ/opt_design/data_cdf_lcox")
# datastrat4=np.loadtxt(Strat4,skiprows=1)
# ax[0].plot(datastrat4[:,0],datastrat4[:,1],color='black')
# ax[1].plot(datastrat4[:,0],datastrat4[:,1],color='lightgray',linewidth=0.5)
# ax[2].plot(datastrat4[:,0],datastrat4[:,1],color='lightgray',linewidth=0.5)
# 
# Strat5=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/UQ/opt_design/data_cdf_lcox")
# datastrat5=np.loadtxt(Strat5,skiprows=1)
# ax[0].plot(datastrat5[:,0],datastrat5[:,1],color='lightgray',linewidth=0.5)
# 
# ax[2].plot(datastrat5[:,0],datastrat5[:,1],color='lightgray',linewidth=0.5)
# 
# Strat8=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/UQ/opt_design/data_cdf_lcox")
# datastrat8=np.loadtxt(Strat8,skiprows=1)
# ax[0].plot(datastrat8[:,0],datastrat8[:,1],color='lightgray',linewidth=0.5)
# 
# ax[2].plot(datastrat8[:,0],datastrat8[:,1],color='lightgray',linewidth=0.5)
# 
# Strat6=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/UQ/opt_design/data_cdf_lcox")
# datastrat6=np.loadtxt(Strat6,skiprows=1)
# ax[0].plot(datastrat6[:,0],datastrat6[:,1],color='lightgray',linewidth=0.5)
# ax[1].plot(datastrat6[:,0],datastrat6[:,1],color='lightgray',linewidth=0.5)
# ax[2].plot(datastrat6[:,0],datastrat6[:,1],color='red',linewidth=1)
# 
# Strat7=("/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/UQ/opt_design/data_cdf_lcox")
# datastrat7=np.loadtxt(Strat7,skiprows=1)
# #ax[0].set_ylabel('Cumulative\nprobability',rotation=0,loc='top')
# ax[0].set_yticks([0.62,0.55,0.95])
# ax[1].set_yticks([0.62,0])
# ax[2].set_yticks([0.62,0.7])
# ax[0].plot(datastrat2[:,0],datastrat2[:,1],color='hotpink',linewidth=1)
# ax[1].plot(datastrat2[:,0],datastrat2[:,1],color='hotpink',linewidth=1)
# ax[2].plot(datastrat2[:,0],datastrat2[:,1],color='hotpink',linewidth=1)
# ax[0].plot(datastrat7[:,0],datastrat7[:,1],color='lightgray',linewidth=0.5)
# ax[1].plot(datastrat7[:,0],datastrat7[:,1],color='lightgray',linewidth=0.5)
# ax[2].plot(datastrat7[:,0],datastrat7[:,1],color='orange')
# ax[0].plot(datastrat3[:,0],datastrat3[:,1],color='green')
# ax[1].plot(datastrat5[:,0],datastrat5[:,1],color='blue')
# ax[1].plot(datastrat8[:,0],datastrat8[:,1],color='purple')
# 
# 
# ax[0].spines['right'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[2].spines['right'].set_visible(False)
# 
# ax[0].spines['top'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# 
# 
# ax[0].spines['bottom'].set_visible(False)
# ax[0].axes.xaxis.set_visible(False)
# 
# ax[1].spines['bottom'].set_visible(False)
# ax[1].axes.xaxis.set_visible(False)
# 
# #plt.legend(['Reference strategy','Grid\'s buying strategy','Grid\'s selling strategy','Temperature dependent \nheat pump strategy','Partially charging strategy','Peak shaving strategy','Forecast strategy'])
# 
# 
# 
# plt.xlabel('LCOX mean [€/MWh]')
# #plt.ylabel('Cumulative \nprobability',rotation=0,loc='top')
# 
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# #plt.yticks([0,1],["0","1"])
# 
# 
# maxx = max(max(datastrat8[:,0]),max(datastrat7[:,0]),max(datastrat6[:,0]),max(datastrat5[:,0]),max(datastrat4[:,0]),max(datastrat3[:,0]),max(datastrat2[:,0]))
# minx= min(min(datastrat8[:,0]),min(datastrat7[:,0]),min(datastrat6[:,0]),min(datastrat5[:,0]),min(datastrat4[:,0]),min(datastrat3[:,0]),min(datastrat2[:,0]))
# 
# #plt.yticks([0.10,0.13,0.22,0.24,0.28,0.77,1],[0.10,0.13,0.22,0.24,0.28,0.77,1])
# plt.xticks([int(round(maxx)),int(round(minx)),550],[int(round(maxx)),int(round(minx)),550])

#plt.set_yticks([0,0.10,0.12,0.17,0.24,0.36,0.76,1], minor=False)
#ax.set_yticks([0.3, 0.55, 0.7], minor=True)



## LCOX reference Gray + Normal A METTRE EN BONNE COULEURS
fig, ax = plt.subplots()

result_dir = ['/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/Test','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/Test','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/Test','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/Test_new','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/Test']
for i in result_dir:
    if i== '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y02,y12,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y02, y12, 'hotpink',linewidth=1.5)
        #plt.text(-6.5, 0.1, 'Reference \n Strategy', horizontalalignment='center', verticalalignment='center',color='gray')
        #maxx2=round(max(y0),2)
        #minx2 =round(min(y0),2)
        #maxy2=round(max(y1),2)
        #miny2 =round(min(y1),2)
    
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/TEST2':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'dimgrey',linewidth=0.5)
        
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/TEST2':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'dimgrey',linewidth=0.5)
        
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/TEST2':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'dimgrey',linewidth=0.5)
        
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/TEST2':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'dimgrey',linewidth=0.5)
        
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/TEST2':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'dimgrey',linewidth=0.5)
    
    elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/Test_new':
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'orange',linewidth=1.5)
        maxx3=round(max(y0))
        minx3 =round(min(y0))
        maxy3=round(max(y1),1)
        miny3 =round(min(y1),1)
    
    
    elif i=='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h':
        print(i)
        my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
        y0,y1,x = my_opt_plot.get_fitness_population(i)
        ax.plot(y0, y1, 'red',linewidth=1.5)
        maxx1=round(max(y0))
        minx1 =round(min(y0))
        maxy1=round(max(y1),1)
        miny1 =round(min(y1),1)
        #ax.scatter([max(y0)],[min(y1)], edgecolors='green',facecolors='none')
        #ax.scatter([min(y0)],[max(y1)], edgecolors='orange',facecolors='none')
        #ax.annotate('Sobol indices:\n'+smalldev1text + ': '+str(smalldev1number) + '\n' + smalldev2text + ': '+str(smalldev2number) + '\n' + smalldev3text + ': '+str(smalldev3number) , (max(y0),min(y1)), (maxx1,50),arrowprops=dict(color='green', width=0.5,headwidth=5),horizontalalignment='right',color='royalblue',size='small')
        #ax.annotate('Sobol indices:\n'+smallmean1text + ': '+str(smallmean1number) + '\n' + smallmean2text + ': '+str(smallmean2number) + '\n' + smallmean3text + ': '+str(smallmean3number) , (min(y0),max(y1)), (530,maxy1),arrowprops=dict(color='orange', width=0.5,headwidth=5),horizontalalignment='left',color='royalblue',size='small')


maxx2=round(max(y02))
minx2 =round(min(y02))
maxy2=round(max(y12),1)
miny2 =round(min(y12),1)
#ax.annotate('Sobol indices:\n'+smalldev1text2 + ': '+str(smalldev1number2) + '\n' + smalldev2text2 + ': '+str(smalldev2number2) + '\n' + smalldev3text2 + ': '+str(smalldev3number2) , (max(y02),min(y12)), (max(y02),50),arrowprops=dict(color='gray', width=0.5,headwidth=5),horizontalalignment='right',color='gray',size='small')
#ax.annotate('Sobol indices:\n'+smallmean1text2 + ': '+str(smallmean1number2) + '\n' + smallmean2text2 + ': '+str(smallmean2number2) + '\n' + smallmean3text2 + ': '+str(smallmean3number2) , (min(y02),max(y12)), (550,max(y12)),arrowprops=dict(color='gray', width=0.5,headwidth=5),horizontalalignment='left',color='gray',size='small')


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
ax.set_ylabel('LCOX \n standard \n deviation \n[€/MWh]',rotation=0,loc='top')
ax.set_xlabel('LCOX mean [€/MWh]',rotation=0)
#ax[0].legend()


#plt.setp(ax, xticks=[maxx2,minx2],yticks=[miny2,maxy2])
plt.setp(ax, xticks=[minx2,maxx2,minx1,maxx1,minx3,maxx3],yticks=[miny2,maxy2,maxy1,miny1,miny3,maxy3])

ax.xaxis.set_tick_params(labelsize=8.5)
    
# #    
# 
### CHARACTERISTICS in Subplot

# 
# # # 
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y01,y11,x1 = my_opt_plot.get_fitness_population('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test')
#  
# #  #### Design charactéristiqyes
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0,y1,x = my_opt_plot.get_fitness_population('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h')
# fig, ax = plt.subplots(6,1)
# 
# ax[0].plot(y0, x[0], 'red', label='PV',linewidth=1)
# ax[1].plot(y0, x[1], 'red', label='Bat',linewidth=1)
# ax[2].plot(y0, x[2], 'red', label='HP',linewidth=1) # to plot one design charact (order of design space file) in function lcox mean
# ax[3].plot(y0, x[3],'red', label='Tank',linewidth=1)
# ax[4].plot(y0, x[4], 'red', label='Tank',linewidth=1)
# ax[5].plot(y0, x[5], 'red', label='Tank',linewidth=1)
# 
# 
# 
# ax[0].plot(y01, x1[0], 'hotpink', label='PV',linewidth=1)
# ax[1].plot(y01, x1[1], 'hotpink', label='Bat',linewidth=1)
# ax[2].plot(y01, x1[2], 'hotpink', label='HP',linewidth=1) # to plot one design charact (order of design space file) in function lcox mean
# ax[3].plot(y01, x1[3], 'hotpink', label='Tank',linewidth=1)
# #ax[2,0].plot(y0, x[4], 'royalblue', label='autre',linewidth=1)
# #ax[2,1].plot(y0, x[5], 'royalblue', label='autre2.0',linewidth=1)
# #ax[2,1].axis('off')
# 
# maxy10= round(max(x1[0]),1)
# maxy11= round(max(x1[1]),1)
# maxy12= round(max(x1[2]),1)
# maxy13= round(max(x1[3]),1)
# 
# miny10= round(min(x1[0]),1)
# miny11= round(min(x1[1]),1)
# miny12= round(min(x1[2]),1)
# miny13= round(min(x1[3]),1)
# 
# maxx01=int(round(max(y01)))
# minx01=int(round(min(y01)))
# 
# 
# ax[0].spines['right'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[2].spines['right'].set_visible(False)
# ax[3].spines['right'].set_visible(False)
# ax[4].spines['right'].set_visible(False)
# ax[5].spines['right'].set_visible(False)
# #ax[2,1].spines['right'].set_visible(False)
# #ax[2,0].spines['right'].set_visible(False)
# #ax(5).Position(1) = 0.5-ax(5).Position(3)/2;
# 
# ax[0].spines['top'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[3].spines['top'].set_visible(False)
# ax[4].spines['top'].set_visible(False)
# ax[5].spines['top'].set_visible(False)
# #ax[2,1].spines['top'].set_visible(False)
# #ax[2,0].spines['top'].set_visible(False)
# 
# ax[0].spines['bottom'].set_visible(False)
# ax[0].axes.xaxis.set_visible(False)
# 
# ax[1].spines['bottom'].set_visible(False)
# ax[1].axes.xaxis.set_visible(False)
# 
# ax[2].spines['bottom'].set_visible(False)
# ax[2].axes.xaxis.set_visible(False)
# 
# ax[3].spines['bottom'].set_visible(False)
# ax[3].axes.xaxis.set_visible(False)
# 
# ax[4].spines['bottom'].set_visible(False)
# ax[4].axes.xaxis.set_visible(False)
# 
# 
# #ax[3].spines['bottom'].set_visible(False)
# #ax[3].axes.xaxis.set_visible(False)
# 
# #ax[1,0].spines['bottom'].set_visible(False)
# #ax[1,0].axes.xaxis.set_visible(False)
# 
# #ax[1,1].spines['bottom'].set_visible(False)
# #ax[1,1].axes.xaxis.set_visible(False)
# 
# #
# #plt.setp(ax[-1, :], xlabel='LCOX mean [€/MWh]')
# #plt.setp(ax[1, 1], xlabel='LCOX mean [€/MWh]')
# #     plt.setp(ax[0, 0],  ylabel='Photovoltaic capacity [kWp]')
# #     plt.setp(ax[0, 1],  ylabel='Heat pump capacity [kWth]')
# #     plt.setp(ax[1, 1],  ylabel='Thermal storage [L]')
# #     plt.setp(ax[1, 0],  ylabel='Battery capacity [kWh]')
# 
# i=0
# for ax in fig.axes:
#     plt.sca(ax)
#     if i ==0:
#         plt.ylabel('\n\nPhotovoltaic\ncapacity\n[$kW_p$]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         #ax.yaxis.set_label_coords(-0.28,0.5)
#         maxy=round(max(x[0]),1)
#         miny =round(min(x[0]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         #plt.xticks([maxx,minx],[maxx,minx])
#         plt.xticks([maxx,minx,minx01,maxx01],[maxx,minx,minx01,maxx01])
#       
#         plt.yticks([maxy,miny10],[maxy,miny10])
#         #plt.yticks([maxy,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([maxy,maxy,miny10],[maxy,maxy,miny10])
#        
#         
#     
#     if i==1:
#         plt.ylabel('Battery\ncapacity\n[$kWh$]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         maxy=round(max(x[1]),1)
#         miny =round(min(x[1]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         #plt.yticks([maxy,miny],[maxy,miny])
#         plt.yticks([miny,maxy11],[miny,maxy11])
#         if maxy==miny:
#             plt.yticks([maxy11,miny11,maxy],[maxy11,miny11,maxy])
#     if i==2:
#         plt.ylabel('Heat pump\ncapacity\n[$kW_{th}$]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         maxy=round(max(x[2]),1)
#         miny =round(min(x[2]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         #plt.yticks([maxy,miny],[maxy,miny])
#         plt.yticks([maxy,miny,maxy12],[maxy,miny,maxy12])
#         if maxy==miny:
#             plt.yticks([maxy,maxy12],[maxy,maxy12])
#         
#     
#         
#     if i==3:
#         plt.ylabel('Thermal\nstorage\ncapacity\n[$l$]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         #ax.yaxis.set_label_coords(-0.05,0.3)
#         maxy=round(max(x[3]),1)
#         miny =round(min(x[3]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         #plt.xlabel('LCOX mean [€/MWh]')
#         #plt.xticks([maxx,minx],[maxx,minx])
#         print([maxx,minx,minx01,maxx01])
#         plt.xticks([maxx,minx,minx01,maxx01],[maxx,minx,minx01,maxx01])
#         #plt.yticks([maxy,miny],[maxy,miny])
#         plt.yticks([miny,maxy,maxy13],[miny,maxy,maxy13])
#         plt.xlabel('LCOX mean [$€/MWh$]')
#         if maxy==miny:
#             plt.yticks([maxy,maxy13],[maxy,maxy13])
#       
#         
#     if i==4:
#         plt.ylabel('Low\npower\nlimit\n[%]',rotation=0,horizontalalignment ='right')
#         maxy=round(max(x[4]),1)
#         miny =round(min(x[4]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.xticks([maxx,maxx01,minx,minx01],[maxx,maxx01,minx,minx01])
#         plt.yticks([maxy,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([miny,0],[miny,0])
#         #.subplots_adjust(bottom=0.1, right=3, top=0.9)
# #         
#     if i==5:
#         plt.ylabel('High\npower\nlimit\n[$W$]',rotation=0,horizontalalignment ='right')
#         maxy=round(max(x[5]),1)
#         miny =round(min(x[5]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.xticks([maxx,maxx01,minx,minx01],[maxx,maxx01,minx,minx01])
#         plt.yticks([maxy,miny],[maxy,miny]) 
#     i=i+1
#         
# 
# fig.subplots_adjust(hspace=0.05)
# fig.align_labels()
# plt.tight_layout()
# plt.xticks([maxx,minx01,maxx01],[maxx,minx01,maxx01])
# plt.xlabel('LCOX mean [$€/MWh$]')
# #    
# # 
# 
# # 
# # # ##### Courbes de SOBOL designs
# case='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/'
# filenames = ['UQ/newpol2sample_0/full_pce_order_2_[\'lcox\']_Sobol_indices', 'UQ/newpol2sample_1/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_2/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_3/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_4/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_5/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_6/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_7/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_8/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_9/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_10/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_11/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_12/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_13/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_14/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_15/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_16/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_17/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_18/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices']
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0,y1,x = my_opt_plot.get_fitness_population(case+'ROB/Test_new')
# 
# file_in = list()
# x = 0
# elec_cost=np.zeros(20)
# elec_load=np.zeros(20)
# heat_load=np.zeros(20)
# capex_pv=np.zeros(20)
# capex_ashp=np.zeros(20)
# capex_bat=np.zeros(20)
# capex_wtank=np.zeros(20)
# disc_rate=np.zeros(20)
# for item in filenames:
#     path_to_read = (case+filenames[x])
#     content=open(path_to_read)
#     #print(content)
#     content= content.readlines()
#     #print(content)
#     for line in content:
#         line= line.split()
#         if line[0]=="uq_load_e":
#             elec_load[x]=float(line[2])
#             
#         if line[0]=='uq_load_h':
#             heat_load[x]=float(line[2])
#             
#         if line[0]=='elec_cost':
#             elec_cost[x]=float(line[2])
#             
#         if line[0]== 'capexhp':
#             capex_ashp[x]=float(line[2])
#             
#         if line[0]=='capex_pv':
#             capex_pv[x]=float(line[2])
#             
#         if line[0]=='capex_bat':
#             capex_bat[x]=float(line[2])
#             
#         if line[0]=='capexwtank':
#             capex_wtank[x] = float(line[2])
#             
#         if line[0]=='disc_rate':
#             disc_rate[x]=float(line[2])
#        
#     
#     
#     x += 1
# 
# 
# maxy1=round(max(elec_load),2)
# maxy2=round(max(heat_load),2)
# maxy3=round(max(elec_cost),2)
# maxy4=round(max(capex_ashp),2)
# maxy5=round(max(capex_pv),2)
# 
# miny1=round(min(elec_load),2)
# miny2=round(min(heat_load),2)
# miny3=round(min(elec_cost),2)
# miny4=round(min(capex_ashp),2)
# miny5=round(min(capex_pv),2)
# 
# 
# case='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/'
# filenames = ['UQ/newpol2sample_0/full_pce_order_2_[\'lcox\']_Sobol_indices', 'UQ/newpol2sample_1/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_2/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_3/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_4/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_5/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_6/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_7/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_8/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_9/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_10/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_11/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_12/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_13/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_14/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_15/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_16/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_17/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_18/full_pce_order_2_[\'lcox\']_Sobol_indices','UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices']
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y02,y12,x = my_opt_plot.get_fitness_population(case+'ROB/Test')
# 
# file_in = list()
# x = 0
# elec_cost2=np.zeros(20)
# elec_load2=np.zeros(20)
# heat_load2=np.zeros(20)
# capex_pv2=np.zeros(20)
# capex_ashp2=np.zeros(20)
# capex_bat2=np.zeros(20)
# capex_wtank2=np.zeros(20)
# disc_rate2=np.zeros(20)
# for item in filenames:
#     path_to_read = (case+filenames[x])
#     content=open(path_to_read)
#     #print(content)
#     content= content.readlines()
#     #print(content)
#     for line in content:
#         line= line.split()
#         if line[0]=="uq_load_e":
#             elec_load2[x]=float(line[2])
#             
#         if line[0]=='uq_load_h':
#             heat_load2[x]=float(line[2])
#             
#         if line[0]=='elec_cost':
#             elec_cost2[x]=float(line[2])
#             
#         if line[0]== 'capexhp':
#             capex_ashp2[x]=float(line[2])
#             
#         if line[0]=='capex_pv':
#             capex_pv2[x]=float(line[2])
#             
#         if line[0]=='capex_bat':
#             capex_bat2[x]=float(line[2])
#             
#         if line[0]=='capexwtank':
#             capex_wtank2[x] = float(line[2])
#             
#         if line[0]=='disc_rate':
#             disc_rate2[x]=float(line[2])
#        
#     
#     
#     x += 1
# 
# maxy12=round(max(elec_load2),2)
# maxy22=round(max(heat_load2),2)
# maxy32=round(max(elec_cost2),2)
# maxy42=round(max(capex_ashp2),2)
# 
# 
# miny12=round(min(elec_load2),2)
# miny22=round(min(heat_load2),2)
# miny32=round(min(elec_cost2),2)
# miny42=round(min(capex_ashp2),2)
# 
# 
# fig, ax = plt.subplots(1,2)
# 
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)  
# ax[0].set_ylabel('Total\nSobol\'\nindex',rotation=0,loc='top')
# ax[0].set_xlabel('LCOX mean [€/MWh]',rotation=0)
# ax[0].plot(y0,elec_load,color='blue')
# ax[0].plot(y0,elec_cost,color='green')
# ax[0].plot(y0,heat_load,color='orange')
# #ax[0].plot(y0,capex_pv,color='purple')
# #ax[0].plot(y0,capex_bat,color='black')
# ax[0].plot(y0,capex_ashp,color='red')
# #ax[0].plot(y0,capex_wtank,color='hotpink')
# #ax[0].plot(y0,disc_rate,color='yellow')
# 
# 
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)  
# ax[1].set_ylabel('Total\nSobol\'\nindex',rotation=0,loc='top')
# ax[1].set_xlabel('LCOX mean [€/MWh]',rotation=0)
# ax[1].plot(y02,elec_load2,color='blue')
# ax[1].plot(y02,elec_cost2,color='green')
# ax[1].plot(y02,heat_load2,color='orange')
# #ax[1].plot(y02,capex_pv2)
# #ax[1].plot(y02,capex_bat2)
# ax[1].plot(y02,capex_ashp2,color='red')
# #ax[1].plot(y02,capex_wtank2)
# #ax[1].plot(y02,disc_rate2)
# 
# 
# minx0=int(round(min(y0)))
# maxx0=int(round(max(y0)))
# 
# minx2=int(round(min(y02)))
# maxx2=int(round(max(y02)))
# 
# #plt.setp(ax[0], yticks=[miny1,miny2,miny3,miny4,maxy1,maxy2,maxy3,maxy4])
# #plt.setp(ax[1], yticks=[miny12,miny22,miny32,miny42,maxy12,maxy22,maxy32,maxy42])
# 
# #plt.xticks([minx2,maxx2],[minx2,maxx2])
# #plt.yticks([miny12,miny32,miny42,maxy12,maxy22,maxy32,maxy42],[miny12,miny32,miny42,maxy12,maxy22,maxy32,maxy42])
# plt.setp(ax[0], yticks=[miny1,miny3,miny4,maxy1,maxy2,maxy4,maxy3])
# plt.setp(ax[1], yticks=[miny12,miny32,miny42,maxy12,maxy22,maxy32,maxy42])
# plt.setp(ax[0], xticks=[minx0,maxx0])
# plt.setp(ax[1], xticks=[minx2,maxx2])
# 




#  BAr chartt + LCOX plot
# fig, ax = plt.subplots(1,2)
# 
# smalldev=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/UQ/Test_robustdesign_26may/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldev= smalldev.readlines()
# for line in contentsmalldev:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldev=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldev=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldev=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldev=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldev=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldev=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldev = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldev=float(line[2])
#    
#    
# smallmean=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/UQ/Test_smallmean_26may/full_pce_order_2_lcox_Sobol_indices')
# contentsmallmean= smallmean.readlines()
# for line in contentsmallmean:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmean=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmean=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmean=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmean=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmean=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmean=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmean = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmean=float(line[2])
#    
#    
# #pour avoir sobol strat2   
# smalldev2=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/UQ/Test_smallstdevStrat2/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldev2= smalldev2.readlines()
# for line in contentsmalldev2:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldev2=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldev2=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldev2=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldev2=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldev2=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldev2=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldev2 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldev2=float(line[2])
#         
#         
# smallmean2=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/UQ/Test/full_pce_order_2_lcox_Sobol_indices')
# contentsmallmean2= smallmean2.readlines()
# for line in contentsmallmean2:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmean2=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmean2=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmean2=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmean2=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmean2=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmean2=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmean2 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmean2=float(line[2])
#   
#         
# labels = ['Electricity\ncost', 'Electricity\nload','Heat\nload','CAPEX\nPV','CAPEX\nbattery','CAPEX\nASHP', 'CAPEX\nthermal\ntank', 'Discount\nrate']
# #print([elec_costsmalldev, uq_load_e_smalldev, uq_load_h_smalldev, capexpvsmalldev, capexbatsmalldev,capexhpsmalldev,capexhpsmalldev,capexwtanksmalldev,discratesmalldev])
# men_means = np.around([elec_costsmalldev, uq_load_e_smalldev, uq_load_h_smalldev, capexpvsmalldev, capexbatsmalldev,capexhpsmalldev,capexwtanksmalldev,discratesmalldev],2)
# #men_means = [25, 32, 34, 20, 25,4,5,6]
# women_means = np.around([elec_costsmallmean, uq_load_e_smallmean, uq_load_h_smallmean, capexpvsmallmean, capexbatsmallmean,capexhpsmallmean,capexwtanksmallmean,discratesmallmean],2)
# 
# 
# smalldev={'Electricity cost': elec_costsmalldev,'Electricity load':uq_load_e_smalldev,'Heat load':uq_load_h_smalldev,'CAPEX PV':capexpvsmalldev,'CAPEX battery':capexbatsmalldev,'CAPEX ASHP':capexhpsmalldev, 'CAPEX themal tank':capexwtanksmalldev, 'Discount rate':discratesmalldev }
# smalldev1text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev1number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev1text]
# smalldev2text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev2number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev2text]
# smalldev3text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev3number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev3text]
# 
# 
# 
# 
# smallmean={'Electricity cost': elec_costsmallmean,'Electricity load':uq_load_e_smallmean,'Heat load':uq_load_h_smallmean,'CAPEX PV':capexpvsmallmean,'CAPEX battery':capexbatsmallmean,'CAPEX ASHP':capexhpsmallmean, 'CAPEX themal tank':capexwtanksmallmean, 'Discount rate':discratesmallmean }
# smallmean1text = max(smallmean.items(), key=operator.itemgetter(1))[0]
# smallmean1number = round(max(smallmean.items(), key=operator.itemgetter(1))[1],2)
# del smallmean[smallmean1text]
# smallmean2text = max(smallmean.items(), key=operator.itemgetter(1))[0]
# smallmean2number = round(max(smallmean.items(), key=operator.itemgetter(1))[1],2)
# del smallmean[smallmean2text]
# smallmean3text = max(smallmean.items(), key=operator.itemgetter(1))[0]
# smallmean3number = round(max(smallmean.items(), key=operator.itemgetter(1))[1],2)
# del smallmean[smallmean3text]
# 
# 
# smalldev2={'Electricity cost': elec_costsmalldev2,'Electricity load':uq_load_e_smalldev2,'Heat load':uq_load_h_smalldev2,'CAPEX PV':capexpvsmalldev2,'CAPEX battery':capexbatsmalldev2,'CAPEX ASHP':capexhpsmalldev2, 'CAPEX themal tank':capexwtanksmalldev2, 'Discount rate':discratesmalldev2 }
# smalldev1text2 = max(smalldev2.items(), key=operator.itemgetter(1))[0]
# smalldev1number2 = round(max(smalldev2.items(), key=operator.itemgetter(1))[1],2)
# del smalldev2[smalldev1text2]
# smalldev2text2 = max(smalldev2.items(), key=operator.itemgetter(1))[0]
# smalldev2number2 = round(max(smalldev2.items(), key=operator.itemgetter(1))[1],2)
# del smalldev2[smalldev2text2]
# smalldev3text2 = max(smalldev2.items(), key=operator.itemgetter(1))[0]
# smalldev3number2 = round(max(smalldev2.items(), key=operator.itemgetter(1))[1],2)
# del smalldev2[smalldev3text2]
# 
# 
# smallmean2={'Electricity cost': elec_costsmallmean2,'Electricity load':uq_load_e_smallmean2,'Heat load':uq_load_h_smallmean2,'CAPEX PV':capexpvsmallmean2,'CAPEX battery':capexbatsmallmean2,'CAPEX ASHP':capexhpsmallmean2, 'CAPEX themal tank':capexwtanksmallmean2, 'Discount rate':discratesmallmean2 }
# smallmean1text2 = max(smallmean2.items(), key=operator.itemgetter(1))[0]
# smallmean1number2 = round(max(smallmean2.items(), key=operator.itemgetter(1))[1],2)
# del smallmean2[smallmean1text2]
# smallmean2text2 = max(smallmean2.items(), key=operator.itemgetter(1))[0]
# smallmean2number2 = round(max(smallmean2.items(), key=operator.itemgetter(1))[1],2)
# del smallmean2[smallmean2text2]
# smallmean3text2 = max(smallmean2.items(), key=operator.itemgetter(1))[0]
# smallmean3number2 = round(max(smallmean2.items(), key=operator.itemgetter(1))[1],2)
# del smallmean2[smallmean3text2]
# 
# 
# 
# 
# #print(women_means)
# 
# x = np.arange(8)  # the label locations
# width = 0.45  # the width of the bars
# 
# #fig, ax = plt.subplots()
# #ax.bar(labels,men_means)
# #plt.plot(men_means,women_means)
# rects1 = ax[1].bar(x - width/2, men_means, width, label='Smallest LCOX \nstandard deviation design',fill=False,edgecolor='green')
# rects2 = ax[1].bar(x + width/2, women_means, width, label='Smallest LCOX \nmean design',fill=False,edgecolor='orange')
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1].set_yticklabels([])
# ax[1].set_ylabel('Sobol indices')
# plt.ylabel(' Sobol \n indices',rotation=0,loc='top')
# ax[1].set_xticks(x)
# ax[1].set_xticklabels(labels, rotation=0,fontsize=7)
# ax[1].legend()
# 
# 
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# ax[1].bar_label(rects1, padding=3,fontsize=7)
# ax[1].bar_label(rects2, padding=3,fontsize=7)
# fig.align_labels()
# 
# 
# result_dir = ['/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/TEST','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/ssrex','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/TEST','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/TEST','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/PB_BAT_HP_TEST_STRAT7_17may','/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/TEST']
# 
# 
# for i in result_dir:
#     if i== '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/ssrex':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y02,y12,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y02, y12, 'dimgrey',linewidth=0.5)
#         #plt.text(-6.5, 0.1, 'Reference \n Strategy', horizontalalignment='center', verticalalignment='center',color='gray')
#         #maxx2=round(max(y0),2)
#         #minx2 =round(min(y0),2)
#         #maxy2=round(max(y1),2)
#         #miny2 =round(min(y1),2)
#     
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/TEST2':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#         
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/TEST2':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#         
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/TEST2':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#         
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/TEST2':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#         
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/TEST2':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#     
#     elif i == '/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/PB_BAT_HP_TEST_STRAT7_1':
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'dimgrey',linewidth=0.5)
#     
#     
#     elif i=='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h':
#         print(i)
#         my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
#         y0,y1,x = my_opt_plot.get_fitness_population(i)
#         ax[0].plot(y0, y1, 'royalblue',linewidth=1.5)
#         maxx1=round(max(y0))
#         minx1 =round(min(y0))
#         maxy1=round(max(y1),1)
#         miny1 =round(min(y1),1)
#         ax[0].scatter([max(y0)],[min(y1)], edgecolors='green',facecolors='none')
#         ax[0].scatter([min(y0)],[max(y1)], edgecolors='orange',facecolors='none')
#         ax[0].annotate('Sobol indices:\n'+smalldev1text + ': '+str(smalldev1number) + '\n' + smalldev2text + ': '+str(smalldev2number) + '\n' + smalldev3text + ': '+str(smalldev3number) , (max(y0),min(y1)), (maxx1,50),arrowprops=dict(color='green', width=0.5,headwidth=5),horizontalalignment='right',color='royalblue',size='small')
#         ax[0].annotate('Sobol indices:\n'+smallmean1text + ': '+str(smallmean1number) + '\n' + smallmean2text + ': '+str(smallmean2number) + '\n' + smallmean3text + ': '+str(smallmean3number) , (min(y0),max(y1)), (530,maxy1),arrowprops=dict(color='orange', width=0.5,headwidth=5),horizontalalignment='left',color='royalblue',size='small')
# 
# 
# maxx2=round(max(y02))
# minx2 =round(min(y02))
# maxy2=round(max(y12),1)
# miny2 =round(min(y12),1)
# ax[0].annotate('Sobol indices:\n'+smalldev1text2 + ': '+str(smalldev1number2) + '\n' + smalldev2text2 + ': '+str(smalldev2number2) + '\n' + smalldev3text2 + ': '+str(smalldev3number2) , (max(y02),min(y12)), (max(y02),50),arrowprops=dict(color='gray', width=0.5,headwidth=5),horizontalalignment='right',color='gray',size='small')
# ax[0].annotate('Sobol indices:\n'+smallmean1text2 + ': '+str(smallmean1number2) + '\n' + smallmean2text2 + ': '+str(smallmean2number2) + '\n' + smallmean3text2 + ': '+str(smallmean3number2) , (min(y02),max(y12)), (550,max(y12)),arrowprops=dict(color='gray', width=0.5,headwidth=5),horizontalalignment='left',color='gray',size='small')
# 
# 
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['top'].set_visible(False)
#     
# ax[0].set_ylabel('LCOX \n standard \n deviation \n[€/MWh]',rotation=0,loc='top')
# ax[0].set_xlabel('LCOX mean [€/MWh]',rotation=0)
# #ax[0].legend()
# 
# 
# 
# plt.setp(ax[0], xticks=[minx1,maxx1],yticks=[miny1,maxy1])
# 
# ax[0].xaxis.set_tick_params(labelsize=8.5)
#     
#    



# 
# 
# ## horizontal bar chart
# avoir uncertainty strat reference du design small st dev
# smalldev=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmalldev= smalldev.readlines()
# for line in contentsmalldev:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldev=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldev=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldev=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldev=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldev=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldev=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldev = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldev=float(line[2])
#         
# 
# smalldev={'Electricity cost': elec_costsmalldev,'Electricity load':uq_load_e_smalldev,'Heat load':uq_load_h_smalldev,'CAPEX PV':capexpvsmalldev,'CAPEX battery':capexbatsmalldev,'CAPEX ASHP':capexhpsmalldev, 'CAPEX themal tank':capexwtanksmalldev, 'Discount rate':discratesmalldev }
# smalldev1text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev1number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev1text]
# smalldev2text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev2number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev2text]
# smalldev3text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev3number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev3text]
# smalldev4text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev4number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# 
#    
#  # avoir uncertainty strat buying du design small mean
# smallmeanstrat3=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmallmeanstrat3= smallmeanstrat3.readlines()
# for line in contentsmallmeanstrat3:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat3=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat3=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat3=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat3 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat3=float(line[2])
#         
# 
# smallmeanstrat3={'Electricity cost': elec_costsmallmeanstrat3,'Electricity load':uq_load_e_smallmeanstrat3,'Heat load':uq_load_h_smallmeanstrat3,'CAPEX PV':capexpvsmallmeanstrat3,'CAPEX battery':capexbatsmallmeanstrat3,'CAPEX ASHP':capexhpsmallmeanstrat3, 'CAPEX themal tank':capexwtanksmallmeanstrat3, 'Discount rate':discratesmallmeanstrat3 }
# smallmean1textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean1textstrat3]
# smallmean2textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean2textstrat3]
# smallmean3textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean3textstrat3]
# 
#    
#    
# # avoir uncertainty strat selling du design small st dev  
# smalldevstrat4=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmalldevstrat4= smalldevstrat4.readlines()
# for line in contentsmalldevstrat4:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat4=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat4=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat4=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat4 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat4=float(line[2])
#         
# smalldevstrat4={'Electricity cost': elec_costsmalldevstrat4,'Electricity load':uq_load_e_smalldevstrat4,'Heat load':uq_load_h_smalldevstrat4,'CAPEX PV':capexpvsmalldevstrat4,'CAPEX battery':capexbatsmalldevstrat4,'CAPEX ASHP':capexhpsmalldevstrat4, 'CAPEX themal tank':capexwtanksmalldevstrat4, 'Discount rate':discratesmalldevstrat4 }
# smalldev1textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev1textstrat4]
# smalldev2textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev2textstrat4]
# smalldev3textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev3textstrat4]
# smalldev4textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev4numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev4textstrat4]
# smalldev5textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev5numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
#         
# # Pour avoir uncertainty small mean strat 5
# 
# smallmeanstrat5=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmallmeanstrat5= smallmeanstrat5.readlines()
# for line in contentsmallmeanstrat5:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat5=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat5=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat5=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat5 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat5=float(line[2])
# 
# smallmeanstrat5={'Electricity cost': elec_costsmallmeanstrat5,'Electricity load':uq_load_e_smallmeanstrat5,'Heat load':uq_load_h_smallmeanstrat5,'CAPEX PV':capexpvsmallmeanstrat5,'CAPEX battery':capexbatsmallmeanstrat5,'CAPEX ASHP':capexhpsmallmeanstrat5, 'CAPEX themal tank':capexwtanksmallmeanstrat5, 'Discount rate':discratesmallmeanstrat5 }
# smallmean1textstrat5 = max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean1textstrat5]
# smallmean2textstrat5 = max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean2textstrat5]
# smallmean3textstrat5= max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean3textstrat5]
# smallmean4textstrat5= max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean4numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# 
# 
# ## uncertainties strat 6 small st dev
# smalldevstrat6=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmalldevstrat6= smalldevstrat6.readlines()
# for line in contentsmalldevstrat6:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat6=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat6=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat6=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat6 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat6=float(line[2])
#         
# smalldevstrat6={'Electricity cost': elec_costsmalldevstrat6,'Electricity load':uq_load_e_smalldevstrat6,'Heat load':uq_load_h_smalldevstrat6,'CAPEX PV':capexpvsmalldevstrat6,'CAPEX battery':capexbatsmalldevstrat6,'CAPEX ASHP':capexhpsmalldevstrat6, 'CAPEX themal tank':capexwtanksmalldevstrat6, 'Discount rate':discratesmalldevstrat6 }
# smalldev1textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev1textstrat6]
# smalldev2textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev2textstrat6]
# smalldev3textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev3textstrat6]
# smalldev4textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev4numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# 
# 
# 
# 
# 
# ## uncertainties strat 8 small st dev
# smalldevstrat8=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmalldevstrat8= smalldevstrat8.readlines()
# for line in contentsmalldevstrat8:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat8=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat8=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat8=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat8 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat8=float(line[2])
#         
# smalldevstrat8={'Electricity cost': elec_costsmalldevstrat8,'Electricity load':uq_load_e_smalldevstrat8,'Heat load':uq_load_h_smalldevstrat8,'CAPEX PV':capexpvsmalldevstrat8,'CAPEX battery':capexbatsmalldevstrat8,'CAPEX ASHP':capexhpsmalldevstrat8, 'CAPEX themal tank':capexwtanksmalldevstrat8, 'Discount rate':discratesmalldevstrat8 }
# smalldev1textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev1textstrat8]
# smalldev2textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev2textstrat8]
# smalldev3textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev3textstrat8]
# smalldev4textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev4numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# 
# 
# 
# 
# # Pour avoir uncertainty small mean strat 7
# 
# smallmeanstrat7=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/UQ/newpol2sample_19/full_pce_order_2_[\'lcox\']_Sobol_indices')
# contentsmallmeanstrat7= smallmeanstrat7.readlines()
# for line in contentsmallmeanstrat7:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat7=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat7=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat7=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat7 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat7=float(line[2])
# 
# smallmeanstrat7={'Electricity cost': elec_costsmallmeanstrat7,'Electricity load':uq_load_e_smallmeanstrat7,'Heat load':uq_load_h_smallmeanstrat7,'CAPEX PV':capexpvsmallmeanstrat7,'CAPEX battery':capexbatsmallmeanstrat7,'CAPEX ASHP':capexhpsmallmeanstrat7, 'CAPEX themal tank':capexwtanksmallmeanstrat7, 'Discount rate':discratesmallmeanstrat7 }
# smallmean1textstrat7 = max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean1textstrat7]
# smallmean2textstrat7 = max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean2textstrat7]
# smallmean3textstrat7= max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean3textstrat7]
# smallmean4textstrat7= max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean4numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# 
# 
# # fig, ax = plt.subplots()
# # 
# # print(smalldev)
# # strat2 = smalldev[]
# # strat3 = [smallmean1numberstrat3,smallmean2numberstrat3,smallmean3numberstrat3]
# # 
# # 
# # x = np.arange(3)  # the label locations
# # width = 0.3
# # print(x)
# # rects1 = ax.barh(x- width/2 ,strat2, width,fill=False,edgecolor='hotpink')
# # rects2 = ax.barh(x + width/2, strat3, width,fill=False,edgecolor='green')
# # 
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_yticklabels([])
# # ax.set_ylabel('Sobol indices')
# # plt.ylabel(' Sobol \n indices',rotation=0,loc='top')
# # ax.set_xticks(x)
# # #ax.set_xticklabels(labels, rotation=0,fontsize=7)
# # ax.legend()
# # 
# # 
# # plt.gca().spines['top'].set_visible(False)
# # plt.gca().spines['right'].set_visible(False)
# # ax.bar_label(rects1, padding=3,fontsize=7)
# # ax.bar_label(rects2, padding=3,fontsize=7)
# # fig.align_labels()
# 
# plotdata = pd.DataFrame({
#     "Strat2":[smalldev1number,smalldev2number, smalldev3number,smalldev4number,0],
#     "Strat3":[smallmean3numberstrat3,smallmean2numberstrat3,smallmean1numberstrat3,0,0],   
#     "Strat4":[smalldev2numberstrat4,smalldev1numberstrat4,smalldev4numberstrat4,smalldev3numberstrat4,smalldev5numberstrat4],
#     "Strat5":[smallmean1numberstrat5,smallmean2numberstrat5,smallmean3numberstrat5,smallmean4numberstrat5,0],
#     'Strat8': [smalldev1numberstrat8,smalldev2numberstrat8,smalldev3numberstrat8,smalldev4numberstrat8,0],
#     "Strat6":[smalldev1numberstrat6,smalldev2numberstrat6,smalldev4numberstrat6,smalldev3numberstrat6,0],
#     'Strat7':[smallmean1numberstrat7 ,smallmean2numberstrat7  ,smallmean4numberstrat7,smallmean3numberstrat7,0]
#     }, 
#     index=["Electricity cost", "Electricity load", "Heat load", "CAPEX ASHP", 'CAPEX PV']
# )
# 
# print(plotdata)
# # Define a dictionary mapping variable values to colours:
# plotdata.plot(kind="barh",color=['hotpink','green','black','blue','purple','red','orange'],stacked=True)
# plt.xticks([3.23,1.99,1,0.06,0.55])
# 
# #plt.title("Mince Pie Consumption Study")
# #plt.xlabel("Family Member")
# #plt.ylabel("Total Sobol\' index", loc='center', rotation=0)
# 




####SALKY DIAGRAM

# # avoir uncertainty strat reference du design small st dev
# smalldev=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/UQ/Test_smallstdevStrat2/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldev= smalldev.readlines()
# for line in contentsmalldev:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldev=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldev=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldev=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldev=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldev=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldev=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldev = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldev=float(line[2])
#         
# 
# smalldev={'Electricity cost': elec_costsmalldev,'Electricity load':uq_load_e_smalldev,'Heat load':uq_load_h_smalldev,'CAPEX PV':capexpvsmalldev,'CAPEX battery':capexbatsmalldev,'CAPEX ASHP':capexhpsmalldev, 'CAPEX themal tank':capexwtanksmalldev, 'Discount rate':discratesmalldev }
# smalldev1text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev1number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev1text]
# smalldev2text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev2number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev2text]
# smalldev3text = max(smalldev.items(), key=operator.itemgetter(1))[0]
# smalldev3number = round(max(smalldev.items(), key=operator.itemgetter(1))[1],2)
# del smalldev[smalldev3text]
# 
#    
#  # avoir uncertainty strat buying du design small mean
# smallmeanstrat3=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/UQ/Test/full_pce_order_2_lcox_Sobol_indices')
# contentsmallmeanstrat3= smallmeanstrat3.readlines()
# for line in contentsmallmeanstrat3:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat3=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat3=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat3=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat3=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat3 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat3=float(line[2])
#         
# 
# smallmeanstrat3={'Electricity cost': elec_costsmallmeanstrat3,'Electricity load':uq_load_e_smallmeanstrat3,'Heat load':uq_load_h_smallmeanstrat3,'CAPEX PV':capexpvsmallmeanstrat3,'CAPEX battery':capexbatsmallmeanstrat3,'CAPEX ASHP':capexhpsmallmeanstrat3, 'CAPEX themal tank':capexwtanksmallmeanstrat3, 'Discount rate':discratesmallmeanstrat3 }
# smallmean1textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean1textstrat3]
# smallmean2textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean2textstrat3]
# smallmean3textstrat3 = max(smallmeanstrat3.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat3 = round(max(smallmeanstrat3.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat3[smallmean3textstrat3]
# 
#    
#    
# # avoir uncertainty strat selling du design small st dev  
# smalldevstrat4=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/UQ/Test_smallstdevStrat4/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldevstrat4= smalldevstrat4.readlines()
# for line in contentsmalldevstrat4:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat4=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat4=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat4=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat4=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat4 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat4=float(line[2])
#         
# smalldevstrat4={'Electricity cost': elec_costsmalldevstrat4,'Electricity load':uq_load_e_smalldevstrat4,'Heat load':uq_load_h_smalldevstrat4,'CAPEX PV':capexpvsmalldevstrat4,'CAPEX battery':capexbatsmalldevstrat4,'CAPEX ASHP':capexhpsmalldevstrat4, 'CAPEX themal tank':capexwtanksmalldevstrat4, 'Discount rate':discratesmalldevstrat4 }
# smalldev1textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev1textstrat4]
# smalldev2textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev2textstrat4]
# smalldev3textstrat4 = max(smalldevstrat4.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat4 = round(max(smalldevstrat4.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat4[smalldev3textstrat4]
#         
# # Pour avoir uncertainty small mean strat 5
# 
# smallmeanstrat5=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/UQ/Test/full_pce_order_2_lcox_Sobol_indices')
# contentsmallmeanstrat5= smallmeanstrat5.readlines()
# for line in contentsmallmeanstrat5:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat5=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat5=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat5=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat5=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat5 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat5=float(line[2])
# 
# smallmeanstrat5={'Electricity cost': elec_costsmallmeanstrat5,'Electricity load':uq_load_e_smallmeanstrat5,'Heat load':uq_load_h_smallmeanstrat5,'CAPEX PV':capexpvsmallmeanstrat5,'CAPEX battery':capexbatsmallmeanstrat5,'CAPEX ASHP':capexhpsmallmeanstrat5, 'CAPEX themal tank':capexwtanksmallmeanstrat5, 'Discount rate':discratesmallmeanstrat5 }
# smallmean1textstrat5 = max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean1textstrat5]
# smallmean2textstrat5 = max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean2textstrat5]
# smallmean3textstrat5= max(smallmeanstrat5.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat5 = round(max(smallmeanstrat5.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat5[smallmean3textstrat5]
# 
# 
# ## uncertainties strat 6 small st dev
# smalldevstrat6=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/UQ/Test_robustdesign_26may/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldevstrat6= smalldevstrat6.readlines()
# for line in contentsmalldevstrat6:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat6=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat6=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat6=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat6=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat6 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat6=float(line[2])
#         
# smalldevstrat6={'Electricity cost': elec_costsmalldevstrat6,'Electricity load':uq_load_e_smalldevstrat6,'Heat load':uq_load_h_smalldevstrat6,'CAPEX PV':capexpvsmalldevstrat6,'CAPEX battery':capexbatsmalldevstrat6,'CAPEX ASHP':capexhpsmalldevstrat6, 'CAPEX themal tank':capexwtanksmalldevstrat6, 'Discount rate':discratesmalldevstrat6 }
# smalldev1textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev1textstrat6]
# smalldev2textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev2textstrat6]
# smalldev3textstrat6 = max(smalldevstrat6.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat6 = round(max(smalldevstrat6.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat6[smalldev3textstrat6]
# 
# 
# 
# 
# 
# ## uncertainties strat 8 small st dev
# smalldevstrat8=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/UQ/Test_smallstdevStrat8/full_pce_order_2_lcox_Sobol_indices')
# contentsmalldevstrat8= smalldevstrat8.readlines()
# for line in contentsmalldevstrat8:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smalldevstrat8=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smalldevstrat8=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmalldevstrat8=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmalldevstrat8=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmalldevstrat8 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmalldevstrat8=float(line[2])
#         
# smalldevstrat8={'Electricity cost': elec_costsmalldevstrat8,'Electricity load':uq_load_e_smalldevstrat8,'Heat load':uq_load_h_smalldevstrat8,'CAPEX PV':capexpvsmalldevstrat8,'CAPEX battery':capexbatsmalldevstrat8,'CAPEX ASHP':capexhpsmalldevstrat8, 'CAPEX themal tank':capexwtanksmalldevstrat8, 'Discount rate':discratesmalldevstrat8 }
# smalldev1textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev1numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev1textstrat8]
# smalldev2textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev2numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev2textstrat8]
# smalldev3textstrat8 = max(smalldevstrat8.items(), key=operator.itemgetter(1))[0]
# smalldev3numberstrat8 = round(max(smalldevstrat8.items(), key=operator.itemgetter(1))[1],2)
# del smalldevstrat8[smalldev3textstrat8]
# 
# 
# 
# 
# # Pour avoir uncertainty small mean strat 7
# 
# smallmeanstrat7=open('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/UQ/Test_smallstmean_17may/full_pce_order_2_lcox_Sobol_indices')
# contentsmallmeanstrat7= smallmeanstrat7.readlines()
# for line in contentsmallmeanstrat7:
#     line= line.split()
#     if line[0]=="uq_load_e":
#         uq_load_e_smallmeanstrat7=float(line[2])
#         
#     if line[0]=='uq_load_h':
#         uq_load_h_smallmeanstrat7=float(line[2])
#         
#     if line[0]=='elec_cost':
#         elec_costsmallmeanstrat7=float(line[2])
#         
#     if line[0]== 'capexhp':
#         capexhpsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capex_pv':
#         capexpvsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capex_bat':
#         capexbatsmallmeanstrat7=float(line[2])
#         
#     if line[0]=='capexwtank':
#         capexwtanksmallmeanstrat7 = float(line[2])
#         
#     if line[0]=='disc_rate':
#         discratesmallmeanstrat7=float(line[2])
# 
# smallmeanstrat7={'Electricity cost': elec_costsmallmeanstrat7,'Electricity load':uq_load_e_smallmeanstrat7,'Heat load':uq_load_h_smallmeanstrat7,'CAPEX PV':capexpvsmallmeanstrat7,'CAPEX battery':capexbatsmallmeanstrat7,'CAPEX ASHP':capexhpsmallmeanstrat7, 'CAPEX themal tank':capexwtanksmallmeanstrat7, 'Discount rate':discratesmallmeanstrat7 }
# smallmean1textstrat7 = max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean1numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean1textstrat7]
# smallmean2textstrat7 = max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean2numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean2textstrat7]
# smallmean3textstrat7= max(smallmeanstrat7.items(), key=operator.itemgetter(1))[0]
# smallmean3numberstrat7 = round(max(smallmeanstrat7.items(), key=operator.itemgetter(1))[1],2)
# del smallmeanstrat7[smallmean3textstrat7]




# fig = go.Figure(data=[go.Sankey(
#     node = dict(
#       pad = 20,
#       thickness = 30,
#       line = dict(color = "gray", width = 0.5),
#       #label = ["Reference\nstrategy", "Grid's\nbuying\nstrategy", "Selling", "temp critical", "strat6", "strat7","fefe",'ezd',"Elec cost","elec load","heat load","capex hp","capex pv","disc rate","others"],
#       color = "gray",
#      x = [0.00000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,    1,1,1,1,1,1 ],
#     y = [-0.1, 0.1,0.25,0.4 , 0.8, 1, 0.6,      0.15,0.5,0.72,0.82 ,0.95,1 ]
#     ),
#     link = dict(
#       source = [0,0,0,   1,1,1,    2,2,2,   3,3,3,  4,4,4,    5,5,5,   6,6,6    ], # indices correspond to labels, eg A1, A2, A1, B1, ...
#       target = [8,9,11,  8,9,10,   9,12,13, 8,9,10, 8,9,11,   8,9,10,   8,9,11    ],
#       value = [smalldev1number, smalldev2number, smalldev3number, smallmean1numberstrat3,  smallmean2numberstrat3, smallmean3numberstrat3,smalldev1numberstrat4,smalldev2numberstrat4,smalldev3numberstrat4, smallmean1numberstrat5,smallmean2numberstrat5,smallmean3numberstrat5,smalldev1numberstrat6,smalldev2numberstrat6,smalldev3numberstrat6,smallmean1numberstrat7,  smallmean2numberstrat7, smallmean3numberstrat7,smalldev1numberstrat8,smalldev2numberstrat8,smalldev3numberstrat8],
#       color =['hotpink','hotpink','hotpink', 'green','green','green','black','black','black','blue','blue','blue', 'red','red','red','orange','orange','orange','purple','purple','purple'],
#         
#  ))])
# 
# 
# #fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# 
# plotly.io.write_image(fig, 'output_filenew.pdf', format='pdf')
# 
# fig.show()
# 

plt.show()





