import pandas as pd
import numpy as np
import math
from .info.adlmr_info import adlmr_location as al
from .info.hh101_info import hh101_location as hl101
from .info.testbed_info import testbed_location as tl
from .info.config import config        

def win2feat(windows, sensor_list, max_wd, pf, s_type, data_name):
    s_id={item:num for num, item in enumerate(sensor_list)}
    ws=float(config['ws']); wd_threshold=config['max_wd_threshold']
    if data_name=='adlmr':
        min_loc, max_loc=float(40), float(39*39+28*28)
        # num_sensors
        coord_dict=al
    elif data_name=='hh101':
        min_loc, max_loc=float(32), float(31*31+19*19)
        num_sensors=len(hl101.keys())
        coord_dict=hl101
    else: # testbed
        min_loc, max_loc=float(116), float(10*10+25*25)
        # num_sensors
        coord_dict=tl

    features=[]
    for i in range(windows.shape[0]):
        s_cnt, s_lastfired={item:0 for item in sensor_list}, {item:0 for item in sensor_list}
        feature=[]

        """Time features (of latest event)
            0.  day of week             (dow)   
            1.  hour of day             (hod)  
            2.  seconds past midnight   (spm) 
        """
        begin_t, end_t, middle_t=float(windows[i][0][2]), float(windows[i][-1][2]), float(windows[i][int(len(windows[i])/2)-1][2])
        pts=pd.Timestamp(end_t, unit='s')
        dow_=normalize(pts.dayofweek, 0., 6.)                                   # 1a
        hod_=normalize(pts.hour, 0., 23.)                                       # 1b
        spm_=normalize(pts.hour*3600+pts.minute*60+pts.second, 0., 86399.)      # 1c
        feature+=[dow_, hod_, spm_]

        """Window features
            3.  window duration,                            (wd)
            4.  latest sensor,                              (ls)
            5.  latest sensor location,                     (lsl)  
            6.  most frequent sensor,                       (fs)
            7.  most frequent sensor location,              (fsl)    
            8.  prev most frequent sensor,                  (pfs)
            9.  entropy-based data complexity of window,    (etp)
            10.  activity level change                       (alc)
        """
        wd=end_t-begin_t; 
        
        if max_wd==0.:
            max_wd=max(max_wd, wd)
        elif max_wd!=0:
            if wd/max_wd>wd_threshold:
                max_wd=max_wd
            else:
                max_wd=max(max_wd, wd)
        
        wd_=0 if max_wd==0. else normalize(wd, 0., max_wd)                                    # 2a
        ls_=normalize(s_id[windows[i,-1,0]], 0., num_sensors-1)                                                             # 2b
        x, y=coord_dict[windows[i][-1][0]]; lsl_=normalize(float(x*x+y*y), min_loc, max_loc)                                            # 2c

        maxcnt=-1
        for event in windows[i]:
            s_cnt[event[0]]+=1
            s_lastfired[event[0]]=float(event[2])
        for item in s_id.keys():
            if maxcnt<s_cnt[item]:
                maxcnt=s_cnt[item]
                mfs=item # string (name)
        
        fs_=normalize(s_id[mfs], 0., num_sensors-1)                                                           # 2d
        pfs_=normalize(pf, 0., num_sensors-1)                                                                               # 2e
        pf=s_id[mfs] 
        mfsx, mfsy=coord_dict[mfs]; fsl_=normalize(float(mfsx*mfsx+mfsy*mfsy), min_loc, max_loc)                                       # 2f
        
        # type_cnt={item:0 for item in s_type}; length=2 if data_name=='testbed' else 1
        # for event in windows[i]:
        #     type_cnt[event[0][:length]]+=1

        idx=0
        for k, v in s_cnt.items():
            if v!=0:
                idx+=1
        etp=[0] if idx==1 else [-(v/ws)*math.log((v/ws), math.ceil(ws/idx)) for k, v in s_cnt.items() if v!=0]; etp_=sum(etp)

        # etp=[0 if cnt==0. else -(cnt/ws)*math.log((cnt/ws), math.ceil(ws/len(type_cnt))) for cnt in type_cnt.values() ]; etp_=sum(etp)  # 2g
        alc_=0 if wd==0. else (middle_t-begin_t)/wd                                                                                     # 2h 

        feature+=[wd_, ls_, lsl_, fs_, fsl_, pfs_, etp_, alc_]

        """Sensor features
            11.  count of events                                (coe)                                 
            12.  elapsed time for each sensor since last event  (etl)   
        """
        coe=[normalize(s_cnt[item], 0., ws) for item in s_id.keys()]                                                                    # 3a

        etl=[max(0, end_t-s_lastfired[item]) if float(s_lastfired[item])!=0. else wd for item in s_id.keys()  ]
        etl=etl if wd==0. else [normalize(item, 0., wd) for item in etl]                                                                # 3b

        feature+=coe+etl
        
        features.append(feature)
    # print(max_wd)
    return features, max_wd, pf

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
        # timestamp inaccordance
        return 0
    return (value-min_)/(max_-min_)
