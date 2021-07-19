import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

from .info.adlmr_info import adlmr_location as al
from .info.hh101_info import hh101_location as hl101
from .info.testbed_info import seminar_location as tl
from .info.config import config, feature_name        

def feature_extraction(events, data_name, sensor_list):

    num_sensors=len(sensor_list)
    num_set_features=len(feature_name.keys())-2
    num_total_features=num_set_features+2*num_sensors
    window_size=config['ws']

    # if data_name=='adlmr':
    #     min_loc, max_loc=float(40), float(39*39+28*28)
    #     coord_dict=al
    # elif data_name=='hh101':
    #     min_loc, max_loc=float(32), float(31*31+19*19)
    #     coord_dict=hl101
    # else: # testbed
    #     min_loc, max_loc=float(116), float(10*10+25*25)
    #     coord_dict=tl

    first_dt=datetime.fromtimestamp(float(events[0,2]))
    prevwin1 = prevwin2 = max_duration = 0

    features=[]
    
    sensortimes={item:first_dt-timedelta(days=1) for item in sensor_list} #TODO

    ################################################################

    for event_order, event in enumerate(events):
        feature=np.zeros(num_total_features)

        # window of the latest event
        bucket=[]
        idx=event_order-window_size+1
        while idx<=event_order:
            bucket.append(events[max(0, idx),:])
            idx+=1

        assert len(bucket)==window_size
        assert sum(bucket[-1]==event)==len(event)

        window=np.array(bucket)

        # latest sensor event time
        dt=datetime.fromtimestamp(float(event[2]))
        dt_seconds=dt-dt.replace(hour=0, minute=0, second=0)
        dt_seconds=int(dt_seconds.total_seconds())

        sensortimes[event[0]]=dt

        # A0 (time of the last sensor event in window (hour))
        # A1 (time of the last sensor event in window (seconds))
        # A2 (day of the week for the last sensor event in window)
        feature[0]=int(dt_seconds/3600);    feature[0]/=24.
        feature[1]=dt_seconds;              feature[1]/=86400.
        feature[2]=dt.weekday();            feature[2]/=7.

        lstime=datetime.fromtimestamp(float(window[-1,2])) # newest sensor event
        lstime=lstime-lstime.replace(hour=0, minute=0, second=0)
        lstime = int(lstime.total_seconds())   

        fstime=datetime.fromtimestamp(float(window[0,2])) # oldest sensor event 
        fstime=fstime-fstime.replace(hour=0, minute=0, second=0)
        fstime = int(fstime.total_seconds())  
        
        duration = lstime-fstime if lstime>=fstime else lstime+(86400-fstime)
        max_duration=max(duration, max_duration)

        # A3 (time duration of entire window)
        feature[3] = duration/max_duration if max_duration!=0. else 0

        halftime=datetime.fromtimestamp(float(window[int(window_size/2)-1,2]))
        halftime=halftime-halftime.replace(hour=0, minute=0, second=0)
        halftime=int(halftime.total_seconds())

        halfduration = halftime-fstime if halftime>=fstime else halftime+(86400-fstime)

        if duration==0.0:
            activitychange=0.0
        else:
            activitychange=float(halfduration)/float(duration)

        # A11 (change in activity level between two halves of current window)
        feature[11] = activitychange

        # A4 (time elapsed since previous sensor event)
        sltime=datetime.fromtimestamp(float(window[-2,2]))
        sltime=sltime-sltime.replace(hour=0, minute=0, second=0)
        sltime=int(sltime.total_seconds())

        since_last_sensor = lstime-sltime if lstime>=sltime else lstime+(86400-sltime)
        feature[4]=since_last_sensor/86400.

        # A7 (last sensor event in current window)
        # feature[7] = sensor_list.index(event[0])
        # feature[7]/= float(len(sensor_list))
        feature[7]=0 #EDIT

        # A8 (last sensor location in current window)
        feature[8] = feature[7] 

        # A9 (last motion sensor location in current window)
        scount=np.zeros(num_sensors)
        last=False
        for ri in range(window_size-1, -1, -1):
            sensor=window[ri, 0]
            scount[sensor_list.index(sensor)]+=1
            if sensor[0]=='M' and not last:
                feature[9]=sensor_list.index(sensor)/len(sensor_list)
                last=True
                break
        if not last:
            feature[9]=0.

        numdistinctsensors=0
        # A10 (complexity of window (entropy calculated from sensor counts))
        complexity=0
        for i in range(num_sensors):
            if scount[i]>1:
                ent = float(scount[i]) / float(window_size)
                ent *= np.log2(ent)
                complexity -= float(ent)
                numdistinctsensors+=1
        feature[10] = complexity

        # A5 (dominant sensor (sensor firing most often) for previous window)
        # feature[5] = prevwin1
        # feature[5]/=float(len(sensor_list))
        feature[5] = 0

        # A6 (dominant sensor two windows back)
        # feature[6] = prevwin2
        # feature[6]/=float(len(sensor_list))
        feature[6] = 0

        # prevwin2 = prevwin1
        # maxcount=0
        # for i in range(num_sensors):
        #     if scount[i]>maxcount:
        #         maxcount=scount[i]
        #         prevwin1 = i

        # (NOT used) number of transitions between areas in current window
        # (NOT used) number of distinct sensors in current window

        # A12+ (counts for each sensor in current window)
        for e in window:
            feature[sensor_list.index(e[0])+num_set_features]+=1
        feature[num_set_features:num_set_features+len(sensor_list)]/=float(window_size)

        # A12+N+ (time elasped since each sensor last fired)

        for i in range(num_sensors):
            difftime=dt-sensortimes[sensor_list[i]]

            if difftime.total_seconds() <0 or (difftime.days > 0):
                feature[num_set_features+num_sensors+i]=86400.
            else:
                feature[num_set_features+num_sensors+i]=difftime.total_seconds()
        feature[num_set_features+len(sensor_list):]/=86400.

        features.append(feature)
        
    return features

def sliding_window(events):
    """
        window w_i = {e_(i-ws+1), ... , e_i}:
            e_i is representative of w_i and remainders {e_(i-ws+1), ..., e_(i-1)} explain the context of e_i
        first (ws-1) number of windows: first event is repeated to keep the size of window (ws)
    """
    ws=config['ws']
    windows=[]
    for i in range(events.shape[0]):
        window=events[i-ws+1:i+1,:]
        if i-ws+1<0:
            repeat=np.array([events[0,:] for j in range(ws-i-1)])
            window=np.concatenate((repeat,events[:i+1]), axis=0)
        windows.append(window)
    windows=np.array(windows)
    
    return windows


def normalize(value, min_, max_):
    if max_==0. or max_-min_==0.:
        print("Error: ({}, {}, {}) Denominator is zero. return -1.".format(max_, min_, value))
        return -1
    elif value<0:
        print("normalize NEGATIVE value")
        return 0
    return (value-min_)/(max_-min_)
