import numpy as np
from itertools import combinations

def convert_motion_event_to_activation(episode, sensors, dataset):

    expire = 40 if dataset=="testbed" else 5

    activation = ["true", "ON"]

    start_time, end_time = float(episode[0,2]), float(episode[-1,2])

    active_log = np.zeros((int(end_time)-int(start_time)+1, len(sensors)))

    active_states = {sensor: False for sensor in sensors}
    active_startt = {sensor: 0 for sensor in sensors}

    for i in range(len(episode)):
        s, v, t = episode[i,:3]
        t = float(t)-start_time
        if s[0]!="M": 
            continue

        if v in activation:

            if active_states[s]==False: # False -> True (State 변경)
                active_startt[s]=t
                active_states[s]=True

            else:                       # True -> True (이전 True까지 Update)

                st_idx, ed_idx = int(active_startt[s]), int(t)
                std, edd = active_startt[s]-np.trunc(active_startt[s]), t-np.trunc(t)

                assert 0<=std<1
                assert 0<=edd<1

                active_log[st_idx:ed_idx,sensors.index(s)]=1
                # active_log[st_idx]=1-std
                # active_log[ed_idx]=edd

                active_startt[s]=t
        else:

            if active_states[s]==True:  # True -> False (이전 True까지 Update하고 초기화)
                # active_log[active_startt[s]:t,sensors.index(s)]=1

                st_idx, ed_idx = int(active_startt[s]), int(t)
                std, edd = active_startt[s]-np.trunc(active_startt[s]), t-np.trunc(t)

                assert 0<=std<1
                assert 0<=edd<1

                active_log[st_idx:ed_idx,sensors.index(s)]=1
                # active_log[st_idx]=1-std
                # active_log[ed_idx]=edd

                active_startt[s]=t
                active_states[s]=False

        
        for sensor in sensors:          # True -> ... (True 이후 이벤트가 없는 경우 - Update하고 초기화)
            if active_states[sensor] and t-active_startt[sensor]>expire:

                st_idx = int(active_startt[sensor])
                std = active_startt[sensor]-np.trunc(active_startt[sensor])

                assert 0<=std<1
                

                # active_log[active_startt[sensor]:active_startt[sensor]+expire,sensors.index(sensor)]=1
                active_log[st_idx:st_idx+expire,sensors.index(sensor)]=1
                # active_log[st_idx]=1-std

                active_states[sensor]=False


        if i==len(episode)-1:
            for sensor in sensors:
                if active_states[sensor]:

                    st_idx = int(active_startt[sensor])
                    std = active_startt[sensor]-np.trunc(active_startt[sensor])

                    assert 0<=std<1

                    active_log[st_idx:,sensors.index(sensor)]=1
                    # active_log[st_idx]=1-std

                    # active_log[active_startt[sensor]:,sensors.index(sensor)]=1
                    active_states[sensor]=False

    return active_log

def activation_to_windows(episode, active_log, transition, time_interval):

    start_time = int(float(episode[0,2]))

    l, r = int(float(episode[transition-1,2]))-start_time, int(float(episode[transition,2]))-start_time
    assert l==r, "End of A equals Start of B"

    tw = l

    time_windows = []
    ground_truth = None
    for i in range(0, len(active_log), time_interval):
        if i <= tw < i+time_interval:
            ground_truth = int(i/time_interval)

        window = np.sum(active_log[i:i+time_interval,:], axis=0)
        time_windows.append(window)

    time_windows = np.array(
        time_windows
    ) #/ time_interval

    return time_windows, ground_truth

def calculate_window_relationship(time_windows, sensors, time_interval):

    window_relationships = []

    for i in range(len(time_windows)):

        curr_correlation = np.zeros((len(sensors), len(sensors)))

        window = time_windows[i]

        # active_sensor_num = len([1 for j in range(len(window)) 
        #     if window[j]>time_interval/len(sensors)])
        # if active_sensor_num==0:
        
        if sum(window)<time_interval/len(sensors):
            window_relationships.append(window_relationships[-1] 
                if len(window_relationships)!=0 else curr_correlation.copy()
            )
            continue

        active_sensors = [j for j in range(len(window))]

        pair = list(combinations(active_sensors, 2)) + [(si, si) for si in range(len(sensors))]

        for a, b in pair:
            r, c = max(a, b), min(a, b)
            weight = window[a]*window[b]/(time_interval**2) if r!=c else window[a]/time_interval
            curr_correlation[r, c] = weight

        window_relationships.append(curr_correlation.copy())

    return window_relationships

def scoring_window_change(time_windows, sensors, window_relationships, prev_length):

    scores = []

    for i in range(len(time_windows)):
        window = time_windows[i]
        
        pair = list(combinations([si for si in range(len(sensors))], 2)) + [(si, si) for si in range(len(sensors))]

        # prev_coefficients = window_relationships[max(0, i-prev_length):i]
        # curr_coefficient = window_relationships[i]
        prev_coefficients = window_relationships[max(0, i-prev_length+1):i+1]
        curr_coefficient = window_relationships[min(i+1, len(time_windows)-1)]
        score = 0.

        if len(prev_coefficients)==0:
            scores.append(0)
            continue

        for a, b in pair:
            r, c = max(a, b), min(a, b)
            prev_c = sum([item[r, c] for item in prev_coefficients])/len(prev_coefficients)
            curr_c = curr_coefficient[r, c]
            score += abs(prev_c-curr_c)
        
        scores.append(score)
    
    return scores