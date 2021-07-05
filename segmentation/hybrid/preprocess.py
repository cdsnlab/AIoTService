from datetime import datetime
import numpy as np
import pandas as pd

def preprocess(raw_data):
    activity=''
    events=[]
    begin_flag=False # to check interleave
    trans_indices=[]
    activity_set=set()
    for i, line in enumerate(raw_data):
        single_event=[]
        f_info=line.decode().split()
        try:
            if f_info[2][0]=='P':
                continue
            if f_info[2]=='AD1-A':
                name='A1'
            elif f_info[2]=='AD1-B':
                name='A2'
            elif f_info[2]=='AD1-C':
                name='A3'
            else:
                name=f_info[2][0]+str(int(f_info[2][1:]))
            single_event.append(name) # 1. sensor
            single_event.append(str(np.array(f_info[3]))) # 2. value
            if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                f_info[1] = f_info[1] + '.000000'
            timestamp=datetime.timestamp(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                "%Y-%m-%d%H:%M:%S.%f"))
            single_event.append(float(timestamp)) # 3. timestamp

            if len(f_info) != 4:  # if activity exists
                trans_indices.append(len(events))
                activity_set.add(f_info[4].strip())
            #     des = str(' '.join(np.array(f_info[4:])))
            #     if 'begin' in des:
            #         if not begin_flag:
            #             begin_flag=True
            #             activity = des.split('=')[0].strip()
            #             single_event.append(activity)
            #         else:
            #             single_event.append(activity)
            #     elif 'end' in des:
            #         if begin_flag:
            #             begin_flag=False
            #             single_event.append(activity)
            #             activity = ''
            #         else:
            #             single_event.append(activity)
            #     else:
            #         continue
            # else:
            #     single_event.append(activity)
            events.append(single_event)
        except IndexError:
            print("{} {}".format(i, line))
    
    return events, trans_indices, activity_set

def preprocess_adlmr(raw_data):
    activity = ''
    events=[]
    current_task={'1':'0', '2':'0'}
    trans_indices=[]
    activity_set=set()
    for i, line in enumerate(raw_data):
        single_event=[]
        f_info=line.decode().split()
        try:
            name=f_info[2][0]+str(int(f_info[2][1:]))
            single_event.append(name) # 1. sensor
            single_event.append(str(np.array(f_info[3]))) # 2. value
            if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                f_info[1] = f_info[1] + '.000000'
            timestamp=datetime.timestamp(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                "%Y-%m-%d%H:%M:%S.%f"))
            single_event.append(float(timestamp)) # 3. timestamp (mili-sec)

            if len(f_info)!=4:
                if len(f_info)==6:  # if activity exists
                    resident=str(np.array(f_info[4])).strip()
                    task=str(np.array(f_info[5])).strip()
                    activity_set.add(task)
                    if current_task[resident]!=task:
                        current_task[resident]=task
                        trans_indices.append(len(events))
                    single_event.append(resident) # 4. resident_ID
                    single_event.append(task) # 5. task_ID
                elif len(f_info)==8:
                    r1=str(np.array(f_info[4])).strip()
                    t1=str(np.array(f_info[5])).strip()
                    r2=str(np.array(f_info[6])).strip()
                    t2=str(np.array(f_info[7])).strip()
                    repeat_flag=False
                    activity_set.add(t1); activity_set.add(t2)

                    if current_task[r1]!=t1:
                        current_task[r1]=t1
                        trans_indices.append(len(events))
                        repeat_flag=True
                    if current_task[r2]!=t2:
                        current_task[r2]=t2
                        if not repeat_flag:
                            trans_indices.append(len(events))
                    single_event.append(str(r1)+str(r2))    # 4.
                    single_event.append(str(t1)+str(t2))    # 5.
                else:
                    print("? {} {}".format(i, line))
            else:
                print("?? {} {}".format(i, line))

            events.append(single_event)
        except IndexError:
            print("{} {}".format(i, line))
    
    return events, trans_indices, activity_set