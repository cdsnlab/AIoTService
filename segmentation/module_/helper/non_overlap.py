import pandas as pd
import numpy as np
from datetime import datetime

def nonoverlap_segments(raw_data):
    overlap, events, ssq=[], [], []
    for i, line in enumerate(raw_data):
        single_event=[]
        f_info=line.decode().split()
        try:
            single_event.append(str(np.array(f_info[2]))) # 1. sensor
            single_event.append(str(np.array(f_info[3]))) # 2. value
            if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                f_info[1] = f_info[1] + '.000000'
            timestamp=datetime.timestamp(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                "%Y-%m-%d%H:%M:%S.%f"))
            single_event.append(float(timestamp)) # 3. timestamp

            if len(f_info) != 4:  # if activity exists
                des = str(' '.join(np.array(f_info[4:])))
                activity = des.split(' ')[0].strip()
                if 'begin' in des:
                    if len(overlap)==0: # begin good
                        overlap.append(activity)
                        single_event.append(activity)
                        ssq.append(single_event)
                    else: # fail
                        overlap, ssq=[], []
                elif 'end' in des:
                    if len(overlap)==1 and overlap[0]==activity:
                        single_event.append(activity)
                        ssq.append(single_event)
                        events.append(ssq)
                        overlap, ssq=[], []
            else:
                if len(overlap)==1:
                    single_event.append(activity)
                    ssq.append(single_event)
        except IndexError:
            print("{} {}".format(i, line))
    
    return events