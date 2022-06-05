# fault_tolerant_tracking


This is the code partially written by Pierre Caprasse used for the master thesis to fulfill the graduation requirements of the
mathematical engineering program at UCLouvain.  It is entitled: Fault-tolerant crossroad multi-tracking in urban areas by Pierre Caprasse.  

A data folder has to be added with the AI city dataset 2021 Track 3: City-Scale Multi-Camera Vehicle Track-ing.

irun_MOMCT.py s the file that has to bu runned in order to execute the implemented MOMCT algorithm. Example of a commandline: 

python run_MOMCT.py --METRICS HOTA --PRINT_RESULTS True --error_case 2  --mode loop  --error_per 10  --BENCHMARK S01  --error_cam 1  --detections yolo --feedback