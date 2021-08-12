import numpy as np
from datetime import datetime
# from module_.readText import read_hh
from collections import Counter

# load dataset
directory_hh101="./dataset/hh/hh101/ann.txt"

f=open(directory_hh101, 'r')
txt=f.readlines()

events=[]

for i, line in enumerate(txt):
    event = []
    
    try:
        f_info = line.split()
        # Date, time, sensor, value, (label)

        event.append(f_info[2])
        event.append(f_info[3])
        if not ('.' in str(np.array(f_info[0])) + f_info[1]):
            f_info[1] = f_info[1] + '.000000'
        timestamp=datetime.timestamp(datetime.strptime(f_info[0] + f_info[1], "%Y-%m-%d%H:%M:%S.%f"))
        event.append(float(timestamp))           # 3. timestamp

        if len(f_info) != 4:
            label = str(' '.join(np.array(f_info[4:])))
            if 'begin' in label or 'end' in label:
                # activity = label.split('=')[0].strip()
                # event.append(activity)
                event.append(label)
            else:
                event.append('None')
        else:
            event.append('None')
        
        events.append(event)

    except:
        print("SKIP")

raw_length = len(events)

flag = False # acting: True (Otherwise: False)
activity = 'Idle'
for i, e in enumerate(events):
    if flag: # True -> acting
        events[i][3] = activity
        if 'end' in e[3]: # activity END
            activity = 'Idle'
            flag = False
    else: # False -> not acting
        if 'begin' in e[3]:
            activity = e[3].split('=')[0].strip()
            events[i][3] = activity
            flag = True
        elif 'None' == e[3]:
            events[i][3] = activity
        else:
            print(i, "?")
            break

assert raw_length == len(events)

# print(len(events))
# print(Counter(np.array(events)[:,0]))

events = [e for e in events if e[0][0] in list('MD')]

startindices = [i for i in range(1, len(events)) if events[i][3]!=events[i-1][3]]







# embedding
# calculate correlation