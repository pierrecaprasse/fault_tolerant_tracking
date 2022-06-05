 import os,sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(path,'POST_PROCESS'))

import post_process_lib as lb

tc = 'PB_BAT_HP_TEST_STRAT0' # name case
my_post_process = lb.post_process(tc) 

# 
eval_type = 'ROB' # l15-l26 pareto front plot,comment if not necessary  
result_dir = ['/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test']
# 
LIGHT = False
# 
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# # 
# # 
# # ## 1LCOX
# for i in result_dir:
#      y01,y11,x1 = my_opt_plot.get_fitness_population(i)
# # # # #     print(i)
# # # # #     ### LCOX mean st dev
# # # # #     plt.plot(y0,y1,'royalblue')
# # # # #     plt.xlabel('Mean lcox [€/MWh]')
# # # # #     plt.ylabel('St dev [€/MWh]',rotation=0)
# # # # #     plt.gca().spines['top'].set_visible(False)
# # # # #     plt.gca().spines['right'].set_visible(False)
# # # # #     maxx=round(max(y0),2)
# # # # #     minx =round(min(y0),2)
# # # # #     plt.xticks([maxx,minx],[maxx,minx])
# # # # #     maxy=round(max(y1),2)
# # # # #     miny =round(min(y1),2)
# # # # #     plt.yticks([maxy,miny],[maxy,miny])
# # # # #     
# # # # #   
# # # #   
# # # #   
# # #  #### Design charactéristiqyes
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0,y1,x = my_opt_plot.get_fitness_population('/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h')
# fig, ax = plt.subplots(6,1)
# 
# ax[0].plot(y0, x[0], 'royalblue', label='PV',linewidth=1)
# ax[1].plot(y0, x[1], 'royalblue', label='Bat',linewidth=1)
# ax[2].plot(y0, x[2], 'royalblue', label='HP',linewidth=1) # to plot one design charact (order of design space file) in function lcox mean
# ax[3].plot(y0, x[3], 'royalblue', label='Tank',linewidth=1)
# ax[4].plot(y0, x[4], 'royalblue', label='Tank',linewidth=1)
# ax[5].plot(y0, x[5], 'royalblue', label='Tank',linewidth=1)
# 
# 
# 
# ax[0].plot(y01, x1[0], 'gray', label='PV',linewidth=1)
# ax[1].plot(y01, x1[1], 'gray', label='Bat',linewidth=1)
# ax[2].plot(y01, x1[2], 'gray', label='HP',linewidth=1) # to plot one design charact (order of design space file) in function lcox mean
# ax[3].plot(y01, x1[3], 'gray', label='Tank',linewidth=1)
# #ax[2,0].plot(y0, x[4], 'royalblue', label='autre',linewidth=1)
# #ax[2,1].plot(y0, x[5], 'royalblue', label='autre2.0',linewidth=1)
# #ax[2,1].axis('off')
# 
# 
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
# #ax[3].spines['bottom'].set_visible(False)
# #ax[3].axes.xaxis.set_visible(False)
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
#         plt.ylabel('Photovoltaic\ncapacity\n[kWp]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         #ax.yaxis.set_label_coords(-0.28,0.5)
#         maxy=round(max(x[0]),1)
#         miny =round(min(x[0]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy+0.2,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([maxy,0],[maxy,0])
#        
#         
#     
#     if i==1:
#         plt.ylabel('Battery\ncapacity\n[kWh]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         maxy=round(max(x[1]),1)
#         miny =round(min(x[1]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.yticks([maxy+1,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([maxy+0.2,0],[maxy,0])
#     if i==2:
#         plt.ylabel('Heat pump\ncapacity\n[kWth]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         maxy=round(max(x[2]),1)
#         miny =round(min(x[2]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.yticks([maxy,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([maxy,0],[maxy,0])
#         
#     
#         
#     if i==3:
#         plt.ylabel('Thermal\nstorage\ncapacity\n[L]',rotation=0,horizontalalignment ='right')
#         fig.align_labels()
#         #ax.yaxis.set_label_coords(-0.05,0.3)
#         maxy=round(max(x[3]),1)
#         miny =round(min(x[3]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         #plt.xlabel('LCOX mean [€/MWh]')
#         plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([maxy,0],[maxy,0])
#       
#         
#     if i==4:
#         plt.ylabel('Low\npower\nlimit\n[W]',rotation=0,horizontalalignment ='right')
#         maxy=round(max(x[4]),1)
#         miny =round(min(x[4]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy,miny],[maxy,miny])
#         if maxy==miny:
#             plt.yticks([miny,0],[miny,0])
#         #.subplots_adjust(bottom=0.1, right=3, top=0.9)
# #         
#     if i==5:
#         plt.ylabel('High\npower\nlimit\n[W]',rotation=0,horizontalalignment ='right')
#         maxy=round(max(x[5]),1)
#         miny =round(min(x[5]),1)
#         maxx=int(round(max(y0)))
#         minx =int(round(min(y0)))
#         plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy,miny],[maxy,miny]) 
#     i=i+1
#         
# 
# 
# fig.align_labels()
# plt.tight_layout()
# plt.xticks([maxx,minx],[int(round(maxx)),int(round(minx))])
# 
# # 
# 
# plt.xlabel('LCOX mean [€/MWh]')
#    
   
# # # ## ALL LCOX
# # 
# Strat2='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test/'
# Strat3='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/TEST/'
# Strat4='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/TEST/'
# Strat5='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/TEST/'
# Strat6='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h/'
# Strat7='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/Test_new/'
# Strat8='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/TEST/'
# 
# #content=[Strat2.readlines(),Strat3.readlines(),Strat4.readlines(),Strat5.readlines(),Strat6.readlines(),Strat7.readlines(),Strat8.readlines()]
# 
# 
# 
# 
# #print(y1s8)
# 
# #plt.plot(y0s3,y1s3)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0s2,y1s2,x = my_opt_plot.get_fitness_population(Strat2)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# y0s3,y1s3,x = my_opt_plot.get_fitness_population(Strat3)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# y0s4,y1s4,x = my_opt_plot.get_fitness_population(Strat4)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# y0s5,y1s5,x = my_opt_plot.get_fitness_population(Strat5)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# y0s6,y1s6,x = my_opt_plot.get_fitness_population(Strat6)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0s7,y1s7,x = my_opt_plot.get_fitness_population(Strat7)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# y0s8,y1s8,x = my_opt_plot.get_fitness_population(Strat8)
# my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
# 
# 
# fig, ax = plt.subplots()
# ax.plot(y0s2,y1s2,color='hotpink')
# ax.scatter(y0s2[4], y1s2[4],c='white',edgecolors='hotpink')
# ax.plot(y0s3,y1s3,color='green')
# ax.scatter(y0s3[2], y1s3[2],c='white',edgecolors='green')
# ax.plot(y0s4,y1s4,color='black')
# ax.scatter(y0s4[5], y1s4[5],c='white',edgecolors='black')
# ax.plot(y0s5,y1s5,color='blue')
# ax.scatter(y0s5[2], y1s5[2],c='white',edgecolors='blue')
# ax.plot(y0s8,y1s8,color='purple')
# ax.scatter(y0s8[4], y1s8[4],c='white',edgecolors='purple')
# ax.plot(y0s6,y1s6,color='red')
# ax.scatter(y0s6[6], y1s6[6],c='white',edgecolors='red')
# ax.plot(y0s7,y1s7,color='orange')
# ax.scatter(y0s7[4], y1s7[4],c='white',edgecolors='orange')
# 
# 
# 
# #ax[1].plot(y0s2,y1s2,color='hotpink')
# #ax[1].scatter(y0s2[19], y1s2[19],c='white',edgecolors='hotpink')
# #ax[1].plot(y0s3,y1s3,color='green')
# #ax[1].scatter(y0s3[0], y1s3[0],c='white',edgecolors='green')
# ##ax[0].plt.plot(y0s4,y1s4,color='black')
# #ax[1].plot(y0s5,y1s5,color='blue')
# #ax[1].scatter(y0s5[0], y1s5[0],c='white',edgecolors='blue')
# #ax[1].plot(y0s8,y1s8,color='purple')
# #ax[1].scatter(y0s8[19], y1s8[19],c='white',edgecolors='purple')
# #ax[1].plot(y0s6,y1s6,color='red')
# #ax[1].scatter(y0s6[19], y1s6[19],c='white',edgecolors='red')
# #ax[1].plot(y0s7,y1s7,color='orange')
# #ax[1].scatter(y0s7[0], y1s7[0],c='white',edgecolors='orange')
# 
# 
# 
# #ax[1].plot(y0s2,y1s2,color='hotpink')
# #ax[1].plot(y0s3,y1s3,color='green')
# #ax[0].plt.plot(y0s4,y1s4,color='black')
# #ax[1].plot(y0s5,y1s5,color='blue')
# #ax[1].plot(y0s6,y1s6,color='red')
# #ax[1].plot(y0s7,y1s7,color='orange')
# #ax[1].plot(y0s8,y1s8,color='purple')
# 
# ax.spines['top'].set_visible(False)
# #ax[1].spines['top'].set_visible(False)
# #ax[1].spines['left'].set_visible(False)
# #ax[1].spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #ax[1].spines['right'].set_visible(False)
# #ax[1].axes.xaxis.set_visible(False)
# #ax[1].axes.yaxis.set_visible(False)
# 
# 
# plt.setp(ax, xlabel='LCOX mean [€/MWh]')
# #plt.setp(ax[1], xlabel='LCOX mean [€/MWh]')
# #ax[0].legend(['Reference\nstrategy','Grid\'s buying\nstrategy','Grid\'s selling\nstrategy','Temperature dependent \nheat pump\nstrategy','Partially charging\nstrategy','Peak shaving\nstrategy','Forecast\nstrategy'],bbox_to_anchor=(-0.24,1.14),loc='upper left',ncol=7)
#        
# 
# 
# i=0
# for ax in fig.axes:
#     plt.sca(ax)
#     if i==0:
#         plt.ylabel('LCOX \nstandard \ndeviation \n[€/MWh]',rotation=0,loc='top')
#         #ax.yaxis.set_label_coords(-0.17,0.9)
#     elif i==1:
#         plt.ylabel('LCOX \nstandard \ndeviation \n[€/MWh]',rotation=0,loc='center')
#         ax.yaxis.set_label_coords(-0.17,0.9)
#     i=i+1
# plt.xlabel('LCOX mean [€/MWh]')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# 
# 
#     
# maxx2 = max(y0s2)
# maxx3 = max(y0s3)
# maxx4 = max(y0s4)
# maxx5 = max(y0s5)
# maxx6 = max(y0s6)
# maxx7 =max(y0s7)
# maxx8 = max(y0s8)
# 
# minx2 =min(y0s2)
# minx3 = min(y0s3)
# minx4 = min(y0s4)
# minx5 = min(y0s5)
# minx6 = min(y0s6)
# minx7 = min(y0s7)
# minx8 = min(y0s8)
# 
# maxy2 = max(y1s2)
# maxy3 = max(y1s3)
# maxy4 = max(y1s4)
# maxy5 = max(y1s5)
# maxy6 = max(y1s6)
# maxy7 = max(y1s7)
# maxy8 = max(y1s8)
# 
# miny2 = min(y1s2)
# miny3 = min(y1s3)
# miny4 = min(y1s4)
# miny5 = min(y1s5)
# miny6 = min(y1s6)
# miny7 = min(y1s7)
# miny8 = min(y1s8)
# 
# maxx= int(round(max([maxx2,maxx3,maxx5,maxx6,maxx7,maxx8])))
# maxy= round(max([maxy2,maxy3,maxy5,maxy6,maxy7,maxy8]),1)
# minx= int(round(min([minx2,minx3,minx5,minx6,minx7,minx8])))
# miny= round(min([miny2,miny3,miny5,miny6,miny7,miny8]),1)
# plt.xticks([maxx,minx,int(round(maxx2)),int(round(maxx6)),int(round(maxx4)),int(round(minx4))],[maxx,minx,int(round(maxx2)),int(round(maxx6)),int(round(maxx4)),int(round(minx4))])
# plt.yticks([maxy,miny,(round(miny4,1)),(round(maxy2,1))],[maxy,miny,(round(miny4,1)),(round(maxy2,1))])
# 
# # inset_ax = fig.add_axes([0.4, 0.4, 0.59, 0.5])
# # 
# # axins=inset_ax.plot(y0s2, y1s2, color='hotpink')
# # inset_ax.plot(y0s3, y1s3, color='green')
# # #inset_ax.plot(y0s4, y1s2, color='hotpink')
# # inset_ax.plot(y0s5, y1s5, color='blue')
# # inset_ax.plot(y0s8, y1s8, color='purple')
# # inset_ax.plot(y0s6, y1s6, color='red')
# # inset_ax.plot(y0s7, y1s7, color='orange')
# # inset_ax.set_xlabel('LCOX mean [€/MWh]')
# # inset_ax.set_ylabel('LCOX\nstandard\ndeviation\n[€/MWh]',rotation=0,loc='top')
# # 
# # inset_ax.scatter(y0s2[19], y1s2[19],c='white',edgecolors='hotpink')
# # #inset_ax.text(y0s2[10], y1s2[10],'Reference strategy', color='hotpink')
# # inset_ax.scatter(y0s3[0], y1s3[0],c='white',edgecolors='green')
# # #inset_ax.text(y0s3[8], y1s3[8],'Grid\'s buying strategy', color='green')
# # #inset_ax.text(y0s4[19]+45, y1s4[19]+7,'Grid\'s selling\nstrategy', color='black')
# # inset_ax.plot(y0s5,y1s5,color='blue')
# # inset_ax.scatter(y0s5[0], y1s5[0],c='white',edgecolors='blue')
# # #inset_ax.text(y0s5[19], y1s5[19],'Temperature dependent strategy', color='blue')
# # inset_ax.plot(y0s8,y1s8,color='purple')
# # inset_ax.scatter(y0s8[19], y1s8[19],c='white',edgecolors='purple')
# # #inset_ax.text(y0s8[19], y1s8[19],'Partially\ncharging\nstrategy', color='purple')
# # inset_ax.plot(y0s6,y1s6,color='red')
# # inset_ax.scatter(y0s6[19], y1s6[19],c='white',edgecolors='red')
# # #inset_ax.text(515, 50,'Peak shaving strategy', color='red')
# # #ax[1].plot(y0s7,y1s7,color='orange')
# # inset_ax.scatter(y0s7[0], y1s7[0],c='white',edgecolors="orange")
# # #inset_ax.text(y0s7[10], y1s7[10],'Forecast strategy', color='orange')
# # #mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# # inset_ax.spines['right'].set_visible(False)
# # inset_ax.spines['top'].set_visible(False)
# # inset_ax.set_xticks([maxx,minx])
# # inset_ax.set_yticks([maxy,miny])
# # 
# 
# #print(y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8)
# #max(y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8)
# #maxy=round(max([y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8]),2)
# #miny =round(min(x[0]),2)
# #maxx=round(max(y1s2),2)
# #minx =round(min(y0),2)

   
#plt.grid(True)

# # # # 
# # ### ALL Charteristics
Strat2='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT2/ROB/Test/'
Strat3='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT3/ROB/TEST/'
Strat4='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT4/ROB/TEST/'
Strat5='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT5/ROB/TEST/'
Strat6='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT6/ROB/New_Test25mai_15h/'
Strat7='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT7/ROB/Test_new/'
Strat8='/Users/MaximeVercoutere/Documents/Unif/Memoire/Drive_Thesis/New_rdo/rdo_framework/RESULTS/PB_BAT_HP_TEST_STRAT8/ROB/TEST/'

#content=[Strat2.readlines(),Strat3.readlines(),Strat4.readlines(),Strat5.readlines(),Strat6.readlines(),Strat7.readlines(),Strat8.readlines()]




#print(y1s8)

#plt.plot(y0s3,y1s3)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
y0s2,y1s2,xs2 = my_opt_plot.get_fitness_population(Strat2)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)

y0s3,y1s3,xs3 = my_opt_plot.get_fitness_population(Strat3)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)

y0s4,y1s4,xs4 = my_opt_plot.get_fitness_population(Strat4)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)

y0s5,y1s5,xs5 = my_opt_plot.get_fitness_population(Strat5)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)

y0s6,y1s6,xs6 = my_opt_plot.get_fitness_population(Strat6)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
y0s7,y1s7,xs7 = my_opt_plot.get_fitness_population(Strat7)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)
y0s8,y1s8,xs8 = my_opt_plot.get_fitness_population(Strat8)
my_opt_plot = lb.post_process_opt(my_post_process,LIGHT,eval_type)


y0t= np.arange(20)

fig, ax = plt.subplots(1,2)

#plt.plot(y0t,xs2[0])
#plt.plot(y0t,xs3[0])
#plt.plot(y0t,xs4[0])
#plt.plot(y0t,xs5[0])
#plt.plot(y0t,xs6[0])
#plt.plot(y0t,xs7[0])
#plt.plot(y0t,xs8[0])



# 


# ax[0].plot(y0t, xs2[0],color='hotpink', label='PV',linewidth=1)
# ax[0].scatter(y0t[19], xs2[0,19],c='white',edgecolors='hotpink')
# ax[0].plot(y0t, xs3[0], color='green',label='PV',linewidth=1)
# ax[0].scatter(y0t[0], xs3[0,0],c='white',edgecolors='green')
# ax[0].plot(y0t, xs4[0],color='black', label='PV',linewidth=1)
# ax[0].scatter(y0t[19], xs4[0,19],c='white',edgecolors='black')
# ax[0].plot(y0t, xs5[0],color='blue', label='PV',linewidth=1)
# ax[0].scatter(y0t[0], xs5[0,0],c='white',edgecolors='blue')
# ax[0].plot(y0t, xs8[0], color='purple',label='PV',linewidth=1)
# ax[0].scatter(y0t[19], xs8[0,19],c='white',edgecolors='purple')
# ax[0].plot(y0t, xs6[0], color='red',label='PV',linewidth=1)
# ax[0].scatter(y0t[19], xs6[0,19],c='white',edgecolors='red')
# ax[0].plot(y0t, xs7[0], color='orange',label='PV',linewidth=1)
# ax[0].scatter(y0t[0], xs7[0,0],c='white',edgecolors='orange')



ax[1].plot(y0t, xs2[1],color='hotpink', label='PV',linewidth=1)
#ax[1].scatter(y0t[19], xs2[1,19],c='white',edgecolors='hotpink')
ax[1].plot(y0t, xs3[1], color='green',label='PV',linewidth=1)
#ax[1].scatter(y0t[0], xs3[1,0],c='white',edgecolors='green')
ax[1].plot(y0t, xs4[1],color='black', label='PV',linewidth=1)
#ax[1].scatter(y0t[19], xs4[1,19],c='white',edgecolors='black')
ax[1].plot(y0t, xs5[1],color='blue', label='PV',linewidth=1)
#ax[1].scatter(y0t[0], xs5[1,0],c='white',edgecolors='blue')
ax[1].plot(y0t, xs8[1], color='purple',label='PV',linewidth=1)
#ax[1].scatter(y0t[19], xs8[1,19],c='white',edgecolors='purple')
ax[1].plot(y0t, xs6[1], color='red',label='PV',linewidth=1)
#ax[1].scatter(y0t[19], xs6[1,19],c='white',edgecolors='red')
ax[1].plot(y0t, xs7[1], color='orange',label='PV',linewidth=1)
#ax[1].scatter(y0t[0], xs7[1,0],c='white',edgecolors='orange')
# 
ax[0].plot(y0t, xs2[2],color='hotpink', label='PV',linewidth=1)
#ax[0].scatter(y0t[19], xs2[2,19],c='white',edgecolors='hotpink')
ax[0].plot(y0t, xs3[2], color='green',label='PV',linewidth=1)
#ax[0].scatter(y0t[0], xs3[2,0],c='white',edgecolors='green')
ax[0].plot(y0t, xs4[2],color='black', label='PV',linewidth=1)
#ax[0].scatter(y0t[19], xs4[2,19],c='white',edgecolors='black')
ax[0].plot(y0t, xs5[2],color='blue', label='PV',linewidth=1)
#ax[0].scatter(y0t[0], xs5[2,0],c='white',edgecolors='blue')
ax[0].plot(y0t, xs8[2], color='purple',label='PV',linewidth=1)
#ax[0].scatter(y0t[19], xs8[2,19],c='white',edgecolors='purple')
ax[0].plot(y0t, xs6[2], color='red',label='PV',linewidth=1)
#ax[0].scatter(y0t[19], xs6[2,19],c='white',edgecolors='red')
ax[0].plot(y0t, xs7[2], color='orange',label='PV',linewidth=1)
#ax[0].scatter(y0t[0], xs7[2,0],c='white',edgecolors='orange')


# 
# ax[1].plot(y0t, xs2[3],color='hotpink', label='PV',linewidth=1)
# ax[1].scatter(y0t[19], xs2[3,19],c='white',edgecolors='hotpink')
# ax[1].plot(y0t, xs3[3], color='green',label='PV',linewidth=1)
# ax[1].scatter(y0t[0], xs3[3,0],c='white',edgecolors='green')
# ax[1].plot(y0t, xs4[3],color='black', label='PV',linewidth=1)
# ax[1].scatter(y0t[19], xs4[3,19],c='white',edgecolors='black')
# ax[1].plot(y0t, xs5[3],color='blue', label='PV',linewidth=1)
# ax[1].scatter(y0t[0], xs5[3,0],c='white',edgecolors='blue')
# ax[1].plot(y0t, xs8[3], color='purple',label='PV',linewidth=1)
# ax[1].scatter(y0t[19], xs8[3,19],c='white',edgecolors='purple')
# ax[1].plot(y0t, xs6[3], color='red',label='PV',linewidth=1)
# ax[1].scatter(y0t[19], xs6[3,19],c='white',edgecolors='red')
# ax[1].plot(y0t, xs7[3], color='orange',label='PV',linewidth=1)
# ax[1].scatter(y0t[0], xs7[3,0],c='white',edgecolors='orange')


ax[0].spines['top'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['right'].set_visible(False)


plt.setp(ax[0], xlabel='LCOX mean [€/MWh]')
plt.setp(ax[1], xlabel='LCOX mean [€/MWh]')


i=0
for ax in fig.axes:
    plt.sca(ax)
    if i ==0:
        plt.ylabel('Heat\npump\ncapacity\n[kWth]',rotation=0,loc='top')
        fig.align_labels()
        ax.yaxis.set_label_coords(-0.07,0.9)
        plt.xticks([],[])
        #ax.legend(['Reference\nstrategy','Grid\'s\nbuying\nstrategy','Grid\'s\nselling\nstrategy','Temperature\ndependent \n heat pump\nstrategy','Peak shaving\nstrategy','Forecast\nstrategy','Partially\ncharging\nstrategy'])


        #maxy=round(max(x[0]),2)
        #miny =round(min(x[0]),2)
        #maxx=round(max(y0),2)
        #minx =round(min(y0),2)
        #plt.xticks([maxx,minx],[maxx,minx])
        #plt.yticks([maxy,miny],[maxy,miny])
       
        
    if i==1:
        plt.ylabel('Battery\ncapacity\n[kWh]',rotation=0,loc='top')
        fig.align_labels()
        ax.yaxis.set_label_coords(-0.07,0.9)
        plt.xticks([],[])
        #ax.legend(['Reference\nstrategy','Grid\'s\nbuying\ntrategy','Grid\'s\nselling\nstrategy','Temperature\ndependent\nheat pump\nstrategy','Peak shaving\nstrategy','Forecast\nstrategy','Partially\ncharging\nstrategy'],bbox_to_anchor=(-1.2,1),loc='lower left',ncol=7)
       
#         maxy=round(max(x[2]),2)
#         miny =round(min(x[2]),2)
#         maxx=round(max(y0),2)
#         minx =round(min(y0),2)
#         #plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy,miny],[maxy,miny])
        
    if i==2:
        
        plt.ylabel('Heat pump \n capacity \n [kWth]',rotation=0,loc='top')
        ax.yaxis.set_label_coords(-0.07,0.9)
        plt.xticks([],[])
#         maxy=round(max(x[1]),2)
#         miny =round(min(x[1]),2)
#         maxx=round(max(y0),2)
#         minx =round(min(y0),2)
        plt.xticks([],[])
#         plt.yticks([maxy,miny],[maxy,miny])
        
    if i==3:
        plt.ylabel('Thermal\nstorage\ncapacity [L]',rotation=0)
        ax.yaxis.set_label_coords(-0.11,0.9)
        #ax.legend(['Reference\nstrategy','Grid\'s\nbuying\nstrategy','Grid\'s\nselling\nstrategy','Temperature\ndependent \n heat pump\nstrategy','Peak shaving\nstrategy','Forecast\nstrategy','Partially\ncharging\nstrategy'],bbox_to_anchor=(-1.3,1.01),loc='lower left',ncol=7)
        plt.xticks([],[])
#         fig.align_labels()
#         maxy=round(max(x[3]),2)
#         miny =0#round(min(x[3]),2)
#         maxx=round(max(y0),2)
#         minx =round(min(y0),2)
#         plt.xticks([maxx,minx],[maxx,minx])
#         plt.yticks([maxy,miny],[maxy,miny])
#     
    i=i+1

    

    

plt.xlabel('LCOX mean [€/MWh]')
#plt.ylabel('Photovoltaic \ncapacity\n[kWp]',rotation=0,loc='top')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.legend(['Reference\nstrategy','Grid\'s\nbuying\nstrategy','Grid\'s\nselling\nstrategy','Temperature\ndependent \nheat pump\nstrategy','Partially\ncharging\nstrategy','Peak shaving\nstrategy','Forecast\nstrategy'],bbox_to_anchor=(-1.3,0.99),loc='lower left',ncol=7)


# #     
# # maxx2 = max(y0s2)
# maxx3 = max(y0s3)
# maxx4 = max(y0s4)
# maxx5 = max(y0s5)
# maxx6 = max(y0s6)
# maxx7 =max(y0s7)
# maxx8 = max(y0s8)
# 
# minx2 =min(y0s2)
# minx3 = min(y0s3)
# minx4 = min(y0s4)
# minx5 = min(y0s5)
# minx6 = min(y0s6)
# minx7 = min(y0s7)
# minx8 = min(y0s8)
# 
# maxy2 = max(y1s2)
# maxy3 = max(y1s3)
# maxy4 = max(y1s4)
# maxy5 = max(y1s5)
# maxy6 = max(y1s6)
# maxy7 = max(y1s7)
# maxy8 = max(y1s8)
# 
# miny2 = min(y1s2)
# miny3 = min(y1s3)
# miny4 = min(y1s4)
# miny5 = min(y1s5)
# miny6 = min(y1s6)
# miny7 = min(y1s7)
# miny8 = min(y1s8)
# 
# maxx= round(max([maxx2,maxx3,maxx5,maxx6,maxx7,maxx8]),2)
# maxy= round(max([maxy2,maxy3,maxy5,maxy6,maxy7,maxy8]),2)
# minx= round(min([minx2,minx3,minx5,minx6,minx7,minx8]),2)
# miny= round(min([miny2,miny3,miny5,miny6,miny7,miny8]),2)
# 

    
    
#print(y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8)
#max(y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8)
#maxy=round(max([y1s2,y1s3,y1s4,y1s5,y1s5,y1s6,y1s7,y1s8]),2)
#miny =round(min(x[0]),2)
#maxx=round(max(y1s2),2)
#minx =round(min(y0),2)


#plt.yticks([maxy,miny],[maxy,miny])
   
plt.show()

#     
# pol_order = 2 # plot sobol indices, comment if not necessary
# objective = 'lcox'
# my_post_process_uq = lb.post_process_uq(my_post_process,pol_order)   
# results_dir = ['newpol2sample_%i' %i for i in range(20)]   
# my_post_process_uq.get_max_sobol(results_dir,objective,threshold=1./27.)



