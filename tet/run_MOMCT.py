from re import S
import sys
import argparse
from Tracking.MOMCT import sort_monoc_dom_color,sort_monoc_nocolor_proj,display_offline, sort_cameras_online
from TrackEval.scripts.run_mot_challenge import eval
import os
from multiprocessing import freeze_support
from TrackEval import trackeval
import numpy as np
import matplotlib.pyplot as plt



sys.path.insert(0, '/Users/djpitchoun/Documents/Etudes/inge civil/master/memoire/')
  
# importing the add and odd_even 
# function
#from module1 import odd_even, add


arg1 = ['display1','display2','feedback','seq_path','phase','max_age','min_hits','mode','iou_threshold','path_train','error_case',
'error_per','error_cam','folder_name','detections']


def parse_args():
    ###
    #SORT AND CLUSTERING CONFIG###
    ###
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT+CLUSTERING demo')
    parser.add_argument('--display1', dest='display1', help='Display online tracker output after sort algorithm',action='store_true')
    parser.add_argument('--display2', dest='display2', help='Display online tracker output after clustering',action='store_true')

    parser.add_argument('--feedback', dest='feedback', help=' activation of feedback  [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=2)
    parser.add_argument("--mode", help="One case of defected detections or loop over it", 
                    type=str, default='loop',choices=["one","loop"] )

    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--path_train", help="path to training data ser", type=str, default='data/AIC21_Track3_MTMCTracking/train/S01')
    parser.add_argument("--error_case", help="case of error", type=int, default=1)

    parser.add_argument("--error_per", help="case of error", type=int, default=20)
    parser.add_argument("--error_cam", help="case of error", type=int, choices=[0,1,2,3,4,6,7,8,9,10], default=2) #0 -> all cam a bit 
    parser.add_argument("--folder_name", help="folder with all data in AI city", type=str, choices=["S01","S02"], default="S01")
    parser.add_argument("--detections", help="Detections as input of the algorithm", type=str, choices=["gt","yolo","masc"], default="gt")

    ##
    ## EVALUATE CONFIG 
    ##
    freeze_support()

    # Command line interface:

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
   # default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    default_metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}

    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs


    for setting in config.keys():

        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    args = parser.parse_args().__dict__

    for setting in args.keys():
        if setting  not in arg1:
            if args[setting] is not None:
                if type(config[setting]) == type(True):
                    if args[setting] == 'True':
                        x = True
                    elif args[setting] == 'False':
                        x = False
                    else:
                        raise Exception('Command line parameter ' + setting + 'must be True or False')
                elif type(config[setting]) == type(1):
                    x = int(args[setting])
                elif type(args[setting]) == type(None):
                    x = None
                elif setting == 'SEQ_INFO':
                    x = dict(zip(args[setting], [None]*len(args[setting])))
                else:
                    x = args[setting]
                config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}


    args = parser.parse_args()
    return args,eval_config,dataset_config,metrics_config


if __name__ == '__main__':
    
    args,eval_config,dataset_config,metrics_config = parse_args()
    args.folder_name = args.BENCHMARK 
    print(args.BENCHMARK)
    mode = args.mode


    file_name =  args.BENCHMARK
    print(" ")
    print("Important parameters")
    print("defected camera:", args.error_cam)
    print("mode:", args.mode)

    if args.mode == "one":
        print("feedback:",args.feedback)
        print("Error percentage deleted:", args.error_per,"%")
        print(" ")
    
        sort_monoc_nocolor_proj(args)
        display_offline(args)
        output_res, output_HOTA, seq_HOTA = eval(eval_config,dataset_config,metrics_config)
        #print(" ")
        #output_HOTA = np.array(output_HOTA)
        #print("HOTA SCORE",output_HOTA[:,0])


    
    if args.mode == "loop":
        
        defected_cam = args.error_cam
        if defected_cam == 0:
             percentages = range(0, 30, 5)
        else: 
            percentages = range(0, 100, 10)
            #percentages = range(0, 95, 5)
        

        #percentages = [0,20]
        cam = str(args.error_cam)


        det_pattern = args.detections
        if det_pattern == "gt":
            outputfile = 'data/HOTA_results/' + file_name + '/' +  'defected_cam_%s'%(cam)
        elif det_pattern == "masc":
            outputfile = 'data/HOTA_results/' + file_name + '/masc/' +  'defected_cam_%s'%(cam)
        elif det_pattern == "yolo":
            outputfile = 'data/HOTA_results/' + file_name + '/yolo/' +  'defected_cam_%s'%(cam)


       
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)



        #FIRST CASE WITH NO FEEDBACK
        args.feedback = False
        
        # with open(os.path.join(outputfile, 'no_feedback.txt'),'w') as out_file:   
            

        #     for per in percentages:
        #         args.error_per = per
        #         print(" ")
        #         print("DEFECTED PERCENTAGE:", args.error_per)
         

        #         sort_monoc_nocolor_proj(args)
        #         display_offline(args)
        #         output_res, output_HOTA, seq_HOTA = eval(eval_config,dataset_config,metrics_config)
        #         if per ==0:
        #              print('%s,%s,%s,%s,%s,%s'%("percentage",seq_HOTA[0],seq_HOTA[1],seq_HOTA[2],seq_HOTA[3],seq_HOTA[4]),file=out_file)  
        #         output_HOTA = np.array(output_HOTA)
        #         d = output_HOTA[:,0]
        #         print('%d,%.2f,%.2f,%.2f,%.2f,%.2f'%(int(per),float(d[0]),float(d[1]),float(d[2]),float(d[3]),float(d[4])),file=out_file)


        # #SECOND CASE WITH FEEDBACK
        args.feedback = True
        

        with open(os.path.join(outputfile, 'feedback.txt'),'w') as out_file:    
            for per in percentages:
                args.error_per = per
                print(" ")
                print("DEFECTED PERCENTAGE:", args.error_per)
         

                sort_monoc_nocolor_proj(args)
                display_offline(args)
                output_res, output_HOTA, seq_HOTA = eval(eval_config,dataset_config,metrics_config)
                if per ==0:
                     print('%s,%s,%s,%s,%s,%s'%("percentage",seq_HOTA[0],seq_HOTA[1],seq_HOTA[2],seq_HOTA[3],seq_HOTA[4]),file=out_file)  
                output_HOTA = np.array(output_HOTA)
                d = output_HOTA[:,0]
                print('%d,%.2f,%.2f,%.2f,%.2f,%.2f'%(int(per),float(d[0]),float(d[1]),float(d[2]),float(d[3]),float(d[4])),file=out_file)
        

        
