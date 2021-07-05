import pandas as pd
import numpy as np

def extract_feature(window, set_sensor, coordinate, prev=0):
    s_id={item:num for num, item in enumerate(set_sensor)}
    s_cnt={item:0 for item in set_sensor}
    s_lastfired={item:0 for item in set_sensor}

    feature=[]

    """1. Time features"""
    begin_t, end_t, middle_t=float(window[0][2]), float(window[-1][2]), float(window[int(len(window)/2)][2])
    pts=pd.Timestamp(begin_t, unit='ms')

    dow=pts.dayofweek
    hod=pts.hour
    spm=pts.hour*3600+pts.minute*60+pts.second
    feature+=[dow, hod, spm]

    """2. Window features"""
    wd=end_t-begin_t
    mrs=s_id[window[-1][0]]
    lslx, lsly=coordinate[window[-1][0]]
    mfs, maxcnt=-1, -1
    for event in window:
        s_cnt[event[0]]+=1
        s_lastfired[event[0]]=float(event[2])
    for item in s_id.keys():
        if maxcnt<s_cnt[item]:
            maxcnt=s_cnt[item]
            mfs=s_id[item]
    for k, v in s_id.items():
        if v==mfs:
            dlx, dly=coordinate[k]
    # edc=?
    # alc=(middle_t-begin_t)/wd
    # numt=?
    # feature+=[wd, mrs, lslx, lsly, mfs, dl, alc]
    feature+=[wd, mrs, lslx, lsly, mfs, dlx, dly]

    """3. Sensor features"""
    coe=[s_cnt[item] for item in s_id.keys()]
    etl=[end_t-s_lastfired[item] if float(s_lastfired[item])!=0. else 0 for item in s_id.keys()]
    feature+=coe+etl
    # for sensor in s_id.keys():
    #     if float(s_lastfired[sensor])==0.:
    #         window_feature.append(0)
    #     else:
    #         elapsed_time=int(end_time-s_lastfired[sensor])
    #         window_feature.append(elapsed_time)
        
    return feature


# Extract "features for CPD"
#   - Time features: 
#       (time of the last event) day of week, hour of day, seconds past midnight
#   - View (Window) features: 
#       view duration, most frequent sensor, complexity, and activity change, last sensor location?
#   - Sensor features: 
#       occurrences, elapsed time since last event

def extract_feature_testbed(window, set_sensor):
    s_id={item:num for num, item in enumerate(set_sensor)}
    s_cnt={item:0 for item in set_sensor}
    s_lastfired={item:0 for item in set_sensor}

    feature=[]

    """1. Time features"""
    begin_t, end_t, middle_t=float(window[0][2]), float(window[-1][2]), float(window[int(len(window)/2)][2])
    pts=pd.Timestamp(begin_t, unit='s')

    dow=pts.dayofweek
    hod=pts.hour
    spm=pts.hour*3600+pts.minute*60+pts.second
    # feature+=[dow, hod, spm]

    """2. Window features"""
    wd=end_t-begin_t
    mrs=s_id[window[-1][0]]
    # lslx, lsly=coordinate[window[-1][0]]
    mfs, maxcnt=-1, -1
    for event in window:
        s_cnt[event[0]]+=1
        s_lastfired[event[0]]=float(event[2])
    for item in s_id.keys():
        if maxcnt<s_cnt[item]:
            maxcnt=s_cnt[item]
            mfs=s_id[item]
    # for k, v in s_id.items():
    #     if v==mfs:
    #         dlx, dly=coordinate[k]
    # edc=?
    # alc=(middle_t-begin_t)/wd
    # numt=?
    # feature+=[wd, mrs, lslx, lsly, mfs, dl, alc]
    # feature+=[wd, mrs, lslx, lsly, mfs, dlx, dly]
    feature+=[wd, mrs, mfs]

    """3. Sensor features"""
    coe=[s_cnt[item] for item in s_id.keys()]
    etl=[(end_t-s_lastfired[item]) if float(s_lastfired[item])!=0. else 0 for item in s_id.keys()]
    feature+=coe+etl
    # for sensor in s_id.keys():
    #     if float(s_lastfired[sensor])==0.:
    #         window_feature.append(0)
    #     else:
    #         elapsed_time=int(end_time-s_lastfired[sensor])
    #         window_feature.append(elapsed_time)
        
    return feature

def normalize_feature(view, num_feat):
    matrix=view.reshape((num_feat,-1))
    col_norm=np.linalg.norm(matrix, axis=0).reshape((1,-1))
    # print(col_norm)
    col_norm=np.where(col_norm==0.,1,col_norm)
    answer=matrix/col_norm[:,None]

    return answer, col_norm
