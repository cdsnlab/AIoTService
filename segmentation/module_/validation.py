import numpy as np

from .featureExtraction import feature_extraction
from .changePointDetection import change_point_detection
from .dataLoader import dataLoader


def check_active_state_order(events, dataset):

    sensor_list = sorted(set(events[:,0]))

    active_dict = {item: [] for item in sensor_list}
    active_flag = {item: False for item in sensor_list}
    active_order = {item: 0 for item in sensor_list}
    active_start = {item: 0. for item in sensor_list}

    if dataset=="hh101":
        on_threshold = 10.
    elif dataset=="adlmr":
        on_threshold = 5.
    elif dataset=="testbed":
        on_threshold = 60.
    else:
        raise ValueError("Wrong dataset")

    for i in range(len(events)):
        sensor, value, timestamp = events[i][:3]
        timestamp = float(timestamp)-float(events[0,2])

        if value in ["ON", "true" ,"OPEN"]:
            if active_flag[sensor]==False:
                active_order[sensor]=i
                active_flag[sensor]=True
                active_start[sensor]=timestamp
        elif value in ["OFF", "false", "CLOSE"]:
            if active_flag[sensor]==True:
                active_dict[sensor].append((active_order[sensor], i))
                active_flag[sensor]=False

        for k, v in active_flag.items(): # True인 상태로 Threshold를 초과했을 때 처리하기
            if v==True and timestamp-active_start[k]>on_threshold:
                active_dict[k].append((active_order[k], i))
                active_flag[k]=False
        
        if i==len(events)-1:
            for k, v in active_flag.items():
                if v==True:
                    active_dict[k].append((active_order[k], i))
                    active_flag[k]=False
    
    return active_dict



def check_active_state(events, dataset):

    if dataset=="testbed":
        threshold_time = 60.
    elif dataset=="adlmr":
        threshold_time = 10.
    elif dataset=="hh101":
        threshold_time = 10.
    else:
        raise ValueError("Wrong dataset.")

    sensor_list = sorted(set(events[:,0]))

    active_dict = {item: [] for item in sensor_list}
    active_flag = {item: False for item in sensor_list}
    active_start = {item: 0. for item in sensor_list}
    active_length = {item: 0 for item in sensor_list}

    start_time = float(events[0,2])

    for i in range(len(events)):
        sensor, value, timestamp = events[i,0], events[i,1], float(events[i,2])-start_time

        # if sensor[0]=="M":
            # if value=='true' or value=='ON': #ON
        if value in ["ON", "true" ,"OPEN"]:
            if active_flag[sensor]==False: #START NEW ACTION
                active_start[sensor]=timestamp
                active_flag[sensor]=True
                active_length[sensor]+=1
            else:
                active_length[sensor]+=1
                print(f"{i} {sensor} ON and ON")

        # elif value=='false' or value=='OFF': #OFF
        elif value in ["OFF", "false", "CLOSE"]:
            if active_flag[sensor]==True: #END ACTION
                active_dict[sensor].append((active_start[sensor], timestamp, active_length[sensor]))
                active_flag[sensor]=False
                active_length[sensor]=0
            else:
                print(f"{i} {sensor} OFF and OFF")

        else:
            print("?", value)

        for k, v in active_start.items(): # True인 상태로 Threshold를 초과했을 때 처리하기
            if active_flag[k]==True and timestamp-active_start[k]>threshold_time:
                active_dict[k].append((active_start[k], active_start[k]+threshold_time))
                active_flag[k]=False
        
        if i==len(events)-1:
            for k, v in active_flag.items():
                if v==True:
                    active_dict[k].append((active_start[k], timestamp))
                    active_flag[k]=False
    
    return active_dict