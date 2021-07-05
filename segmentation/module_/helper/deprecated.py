import pandas as pd
import math
from ..info.hh101_info import hh101_location as hl101

def win2feat_sep(windows, order, set_sensor, max_wd, pf):
    """
        The very first SEP Paper.
    """

    
    s_id={item:num for num, item in enumerate(set_sensor)}

    features=[]
    min_loc, max_loc=float(32), float(19*19+31*31)
    for i in range(windows.shape[0]):
        real_idx=order+i-1

        s_cnt, s_lastfired={item:0 for item in set_sensor}, {item:0 for item in set_sensor}
        feature=[]

        """1. Time features
            a.  day of week (dow, 0-6)   *
            b.  hour of day (hod, 0-23)  *
            c.  seconds past midnight (spm, 0-86399) *
        """
        begin_t, end_t, middle_t=float(windows[i][0][2]), float(windows[i][-1][2]), float(windows[i][int(len(windows[i])/2)-1][2])
        pts=pd.Timestamp(end_t, unit='s')
        dow=pts.dayofweek/6.                               # [0,1]
        hod=pts.hour/23.                                    # [0,1]
        spm=(pts.hour*3600+pts.minute*60+pts.second)/86399.    # [0,1]
        feature+=[dow, hod, spm]

        """2. Window features
            .  window duration,    (wd)
            .  most recent sensor, (mrs)
            .  last sensor location, (lsl)
            .  most frequent sensor, (mfs)
            .  prev most frequent sensor, (pmfs)
            .  most frequent sensor location, (mfsl)
            .  entropy-based data complexity of window, (etp)
            .  activity level change (alc)
        """
        wd=end_t-begin_t; max_wd=max(max_wd, wd)
        if max_wd!=0:
            wd_=wd/max_wd
        else:
            wd_=0.
        mrs=s_id[windows[i][-1][0]]/float(len(set_sensor)-1)

        x, y=hl101[windows[i][-1][0]]
        lsl=(float(x*x+y*y)-min_loc)/(max_loc-min_loc)

        maxcnt=-1
        for event in windows[i]:
            s_cnt[event[0]]+=1
            s_lastfired[event[0]]=float(event[2])
        for item in s_id.keys():
            if maxcnt<s_cnt[item]:
                maxcnt=s_cnt[item]
                mfs=item
        mfs_=s_id[mfs]/float(len(set_sensor)-1)

        if real_idx<0:
            fprev, sprev=pf[0][0]/float(len(set_sensor)-1), pf[0][1]/float(len(set_sensor)-1)
        else:
            fprev, sprev=pf[real_idx][0]/float(len(set_sensor)-1), pf[real_idx][1]/float(len(set_sensor)-1)
            if real_idx+1==len(pf):
                pf.append((pf[real_idx][1], s_id[mfs]))
        pmfs=fprev

        mfsx, mfsy=hl101[mfs]
        mfsl=(float(mfsx*mfsx+mfsy*mfsy)-min_loc)/(max_loc-min_loc)

        idx_=-1; m=0; d=0
        for j, event in enumerate(windows[i]):
            if event[0][0]=='M':
                idx_=j
                m+=1
            else:
                d+=1

        if m==0 or d==0:
            etp=0
        else:
            etp=-(m/30)*math.log((m/30),2)-(d/30)*math.log((d/30), 2)

        if wd==0.:
            alc=0.
        else:
            alc=(middle_t-begin_t)/wd

        feature+=[wd_, mrs, lsl, mfs_, pmfs, mfsl, etp, alc]

        """3. Sensor features
            a.  count of events                                 *
            b.  elapsed time for each sensor since last event   *
        """
        coe=[s_cnt[item]/30. for item in s_id.keys()]

        etl=[max(0, end_t-s_lastfired[item]) if float(s_lastfired[item])!=0. else 0 for item in s_id.keys()]
        if wd!=0.:
            etl=[item/wd for item in etl]

        feature+=coe+etl
        features.append(feature)
        
    return features, max_wd, pf

def win2feat_rulsif(windows, set_sensor, max_wd):
    """
        RuLSIF.
    """
    s_id={item:num for num, item in enumerate(set_sensor)}
    s_cnt_={item:0 for item in set_sensor}
    s_lastfired_={item:0 for item in set_sensor}

    features=[]
    for i in range(windows.shape[0]):
        s_cnt=s_cnt_.copy(); s_lastfired=s_lastfired_.copy()
        feature=[]

        """features
            1. window duration
            2. event time
            3. most dominant (frequent) sensor in current window
            4. number of occurrence of each sensor in window
            5. time each sensor last fired till the end of current
        """
        begin_t, end_t, middle_t=float(windows[i][0][2]), float(windows[i][-1][2]), float(windows[i][int(len(windows[i])/2)][2])
        wd=end_t-begin_t; max_wd=max(max_wd, wd)
        if max_wd!=0:
            wd/=max_wd

        pts=pd.Timestamp(end_t, unit='s'); spm=(pts.hour*3600+pts.minute*60+pts.second)/86399.

        maxcnt=-1
        for event in windows[i]:
            s_cnt[event[0]]+=1
            s_lastfired[event[0]]=float(event[2])
        for item in s_id.keys():
            if maxcnt<s_cnt[item]:
                maxcnt=s_cnt[item]
                mfs=item
        mfs=s_id[mfs]/float(len(set_sensor)-1)

        coe=[s_cnt[item] for item in s_id.keys()]; min_coe, max_coe=min(coe), max(coe)
        coe=[(item-min_coe)/(max_coe-min_coe) for item in coe]

        etl=[max(0, end_t-s_lastfired[item]) if float(s_lastfired[item])!=0. else 0 for item in s_id.keys()]; min_etl, max_etl=min(etl), max(etl)
        if max_wd!=0.:
            etl=[(item-min_etl)/(max_etl-min_etl) for item in etl]

        feature=[wd, spm, mfs]+coe+etl
        features.append(feature)
        
    return features, max_wd 