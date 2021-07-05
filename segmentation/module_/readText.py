from datetime import datetime
from math import remainder
import numpy as np
import pandas as pd

def read_hh(raw_data):
    activity=''
    events=[]
    _cnt = 0
    start=False
    for ln, line in enumerate(raw_data):
        single_event=[]
        f_info=line.decode().split()
        try:
            single_event.append(f_info[2])                  # 1. sensor
            single_event.append(f_info[3])                  # 2. value
            if not ('.' in str(np.array(f_info[0])) + f_info[1]):
                f_info[1] = f_info[1] + '.000000'
            timestamp=datetime.timestamp(datetime.strptime(f_info[0] + f_info[1], "%Y-%m-%d%H:%M:%S.%f"))
            single_event.append(float(timestamp))           # 3. timestamp

            if len(f_info) != 4:  # if activity exists
                des = str(' '.join(np.array(f_info[4:])))
                if 'begin' in des:
                    temp=des.split('=')[0].strip()
                    if _cnt>0:
                        print("[line {}] {} - {} begin - {}".format(ln, activity, temp, activity))
                    _cnt+=1
                    # if start==True:
                    #     print(line, "Break")
                    #     break
                    # else:
                    #     start=True
                    activity = temp
                    single_event.append(activity)
                    
                elif 'end' in des:
                    _cnt-=1
                    if _cnt>0:
                        print("[line {}] ? - {} end - ?".format(ln, activity))
                    # if start==False:
                    #     print(line, "Break")
                    #     break
                    # else:
                    #     start=False
                    single_event.append(activity)
                    activity = ''
                else:
                    # print(ln, line)
                    continue

            else:
                single_event.append(activity)
            events.append(single_event)
        except IndexError:
            print(ln, line)
    
    return events

def read_twor(raw_data):
    activity=''
    events=[]
    begin_flag=False
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
                if 'begin' in des:
                    if begin_flag:
                        print("{} {}".format(i, line))
                    else:
                        begin_flag=True
                    activity = des.split(' ')[0].strip()
                    single_event.append(activity)
                elif 'end' in des:
                    if not begin_flag:
                        print("{} {}".format(i, line))
                    else:
                        begin_flag=False
                    single_event.append(activity)
                    activity = ''
                else:
                    continue
                
            else:
                single_event.append(activity)
            events.append(single_event)
        except IndexError:
            print("{} {}".format(i, line))
    
    return events

def read_adlmr(raw_data):
    """
        Output:
        1. Single stream (A's stream, B's stream)
        2. Group stream
        3. Raw stream (Mixed stream)
    """
    single_stream, group_stream, raw_stream={'1':[], '2':[]}, [], []
    t_single, t_raw={'1':[], '2':[]}, []
    activity={'1':'1', '2':'2'}
    bucket=[]
    group=False
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

            # 4. resident ID # 5. task ID
            if len(f_info)==6:
                rf, tf=str(np.array(f_info[4])), str(np.array(f_info[5]))
                rs, ts='0', '0'
            if len(f_info)==8:
                rf, tf, rs, ts=str(np.array(f_info[4])), str(np.array(f_info[5])), str(np.array(f_info[6])), str(np.array(f_info[7]))

            single_event.append(rf); single_event.append(tf)
            single_event.append(rs); single_event.append(ts)

            if group:
                bucket.append(single_event)
            
            if len(f_info)==7 or len(f_info)==9:
                if 'START' in str(np.array(f_info[-1])):
                    bucket.append(single_event)
                    group=True
                elif 'END' in str(np.array(f_info[-1])):
                    group_stream.append(bucket)
                    bucket=[]
                    group=False

            raw_stream.append(single_event)

            if rf!='0':
                if tf!=activity[rf]:
                    t_single[rf].append(len(single_stream[rf]))
                    t_raw.append(len(raw_stream))
                    activity[rf]=tf
            
            if rs!='0':
                if ts!=activity[rs]:
                    t_single[rs].append(len(single_stream[rs]))
                    t_raw.append(len(raw_stream))
                    activity[rs]=ts

            if str(rf)!='0':
                single_stream[str(rf)].append(single_event)
            if str(rs)!='0':
                single_stream[str(rs)].append(single_event)
        except IndexError:
            print("{} {}".format(i, line))
    
    return single_stream, group_stream, raw_stream, t_single, t_raw

def time_correction(chunk, idx):
    """
        chunk:  the result of concatenation between two different activity array
        idx:    the first event idx of second activity in chunk

        the timestamp of second activity events are corrected and the intra-event time interval of each activity is preserved.
        the time interval between the last event of first activity and the first event of second activity is decided by 
            np.random.randint(a, b)
    """
    sbegts=float(chunk[idx,2])
    fendts=float(chunk[idx-1, 2])
    interval=np.random.randint(10,55)
    for i in range(idx, chunk.shape[0]):
        chunk[i, 2]=str(float(chunk[i, 2])+fendts+interval-sbegts)
    
    return chunk

def create_episodes(task_dict, name_dict):
    """
        task_dict
            key: unique number corresponding to each activity type
            value: list of np array chunk (activity sample)
    """
    episodes, trs, tags=[],[],[]
    for first_ in task_dict.keys():   # choose first activity type
        f_group=task_dict[first_]
        s_group_cand=[item for item in task_dict.keys() if item!=first_]
        for f, f_act in enumerate(f_group):        # choose first activity 
            for second_ in s_group_cand:    # choose second activity type
                s_group=task_dict[second_]
                for s, s_act in enumerate(s_group): # choose second activity
                    episodes.append(np.concatenate((f_act, s_act), axis=0))
                    trs.append(f_act.shape[0])
                    tags.append('{}{}{}{}'.format(name_dict[first_], f, name_dict[second_], s))
    
    return episodes, trs, tags

def create_episodes_intra(task_dict):
    episodes, trs, order = [], [], []
    task_list=list(task_dict.values())
    for i, a_ in enumerate(task_list):
        rest_dict={j:task_list[j] for j in range(len(task_list)) if j!=i}
        rest_list=list(rest_dict.values())
        rest_indices=list(rest_dict.keys())
        for j, b_ in enumerate(rest_list):
            episodes.append(np.concatenate((a_,b_))); 
            trs.append(len(a_)); order.append("{}{}".format(i, rest_indices[j]))
            # episodes.append(np.concatenate((b_,a_))); 
            # trs.append(len(b_)); order.append("{}{}".format(rest_indices[j], i))

    return episodes, trs