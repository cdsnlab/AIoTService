import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

from .info.adlmr import adlmr_location as al
from .info.hh import hh101_location as hl101
from .info.testbed import seminar_location as tl
from .info.config import config, feature_name

DAYHOURS = 24.; DAYSECONDS = 86400.; WEEKDAYS = 7.; HOURSECONDS=3600.
W=config['ws']

def feature_extraction(events, data_name, sensor_list):

    NUMSENSORS, NUMSETFEATURES=len(sensor_list), len(feature_name)-2
    NUMTOTALFEATURES=NUMSETFEATURES+2*NUMSENSORS

    features=[]

    if data_name=='adlmr':
        min_loc, max_loc=float(40), float(39*39+28*28)
        coord_dict=al
    elif data_name=='hh101':
        min_loc, max_loc=float(32), float(31*31+19*19)
        coord_dict=hl101
    elif data_name=='testbed': # testbed
        min_loc, max_loc=float(116), float(10*10+25*25)
        coord_dict=tl

    first_dt = datetime.fromtimestamp(float(events[0,2])) # The very first event of stream
    sensortimes = {item:first_dt-timedelta(days=1) for item in sensor_list} # Initialize Last Fired Time of Every Sensor
    prevwin1 = prevwin2 = 0

    for event_order, event in enumerate(events):

        feature=np.zeros(NUMTOTALFEATURES)

        # Fixed Length Activity Window
        bucket=[]
        idx=event_order-W+1
        while idx<=event_order:
            bucket.append(events[max(0, idx),:])
            idx+=1

        assert len(bucket)==W
        assert sum(bucket[-1]==event)==len(event)

        window=np.array(bucket)

        dt=datetime.fromtimestamp(float(event[2]))
        dt_seconds=dt-dt.replace(hour=0, minute=0, second=0)
        dt_seconds=int(dt_seconds.total_seconds())

        sensortimes[event[0]]=dt

        lstime=datetime.fromtimestamp(float(window[-1,2])) 
        lstime=lstime-lstime.replace(hour=0, minute=0, second=0)
        lstime = int(lstime.total_seconds())

        assert dt_seconds==lstime

        fstime=datetime.fromtimestamp(float(window[0,2])) 
        fstime=fstime-fstime.replace(hour=0, minute=0, second=0)
        fstime = int(fstime.total_seconds())

        halftime=datetime.fromtimestamp(float(window[int(W/2)-1,2]))
        halftime=halftime-halftime.replace(hour=0, minute=0, second=0)
        halftime=int(halftime.total_seconds())

        sltime=datetime.fromtimestamp(float(window[-2,2]))
        sltime=sltime-sltime.replace(hour=0, minute=0, second=0)
        sltime=int(sltime.total_seconds())

        feature[0]=int(dt_seconds/HOURSECONDS)/DAYHOURS # HourOfDay
        feature[1]=dt_seconds/DAYSECONDS # SecondOfDay
        feature[2]=dt.weekday()/WEEKDAYS # DayOfWeek

        duration = lstime-fstime if lstime>=fstime else lstime+(DAYSECONDS-fstime)
        halfduration = halftime-fstime if halftime>=fstime else halftime+(DAYSECONDS-fstime)
        since_last = lstime-sltime if lstime>=sltime else lstime+(DAYSECONDS-sltime)

        feature[3] = duration/DAYSECONDS # Duration
        feature[4] = since_last/DAYSECONDS # ElapsedTimeFromLastEvent
        feature[5] = 0 #halfduration/duration if duration!=0.0 else 0. # ActivityLevelChange

        feature[6] = prevwin1/float(NUMSENSORS) # DominantSensorPrev1W
        feature[7] = prevwin2/float(NUMSENSORS) # DominantSensorPrev2W

        prevwin2 = prevwin1
        scount = np.zeros(NUMSENSORS)
        for i in range(len(window)):
            scount[sensor_list.index(window[i, 0])]+=1
        
        maxcount = 0
        prevwin1 = -1
        for i in range(len(scount)):
            if scount[i]>maxcount:
                maxcount = scount[i]
                prevwin1 = i

        feature[8] = 0#prevwin1/float(NUMSENSORS) # DominantSensorCurrentW
        feature[9] = 0#sensor_list.index(window[0, 0])/float(NUMSENSORS) # FirstSensor
        feature[10] = 0#sensor_list.index(event[0])/float(NUMSENSORS) # LastSensor

        feature[11] = (sum(np.square(coord_dict[event[0]]))-min_loc)/(max_loc-min_loc) # LastSensorLocation

        for ri in range(W-1, -1, -1):
            if window[ri, 0][0]=='M':
                feature[12] = (sum(np.square(coord_dict[window[ri, 0]]))-min_loc)/(max_loc-min_loc) #LastMotionSensorLocation
                break

        feature[13] = 0#(sum(np.square(coord_dict[sensor_list[prevwin1]]))-min_loc)/(max_loc-min_loc) #DominantSensorLocation

        numdistinctsensors=0
        complexity=0
        for i in range(NUMSENSORS):
            if scount[i] >= 1:
                ent = float(scount[i]) / float(W)
                ent *= np.log2(ent)
                complexity -= float(ent)
                numdistinctsensors+=1

        feature[14] = complexity/np.log2(W) # DataComplexity
        feature[15] = 0#numdistinctsensors/float(NUMSENSORS) # DistinctSensors

        for e in window:
            feature[sensor_list.index(e[0])+NUMSETFEATURES]+=0#1

        assert sum(feature[NUMSETFEATURES:NUMSETFEATURES+NUMSENSORS])==0

        feature[NUMSETFEATURES:NUMSETFEATURES+NUMSENSORS]/=float(W)

        for i in range(NUMSENSORS):
            difftime=dt-sensortimes[sensor_list[i]]

            if difftime.total_seconds()<0 or difftime.days>0:
                feature[NUMSETFEATURES+NUMSENSORS+i]=DAYSECONDS
            else:
                feature[NUMSETFEATURES+NUMSENSORS+i]=difftime.total_seconds()

        feature[NUMSETFEATURES+NUMSENSORS:]/=DAYSECONDS

        features.append(feature)
        
    return features

def sliding_window(values, window_size):
    """
        window w_i = {e_(i-ws+1), ... , e_i}:
            e_i is representative of w_i and remainders {e_(i-ws+1), ..., e_(i-1)} explain the context of e_i
        first (ws-1) number of windows: first event is repeated to keep the size of window (ws)
    """

    windows = []
    for i in range(len(values)):
        window=[]
        idx=i-window_size+1
        while idx<=i:
            window.append(values[max(0, idx)])
            idx+=1
        
        assert len(window)==window_size
        assert window[-1]==values[i]

        windows.append(window)
    
    assert len(values)==len(windows)

    return windows


def normalize(value, min_, max_):
    if max_==0. or max_-min_==0.:
        print("Error: ({}, {}, {}) Denominator is zero. return -1.".format(max_, min_, value))
        return -1
    elif value<0:
        print("normalize NEGATIVE value")
        return 0
    return (value-min_)/(max_-min_)
