import numpy as np
from itertools import combinations

def similarity(events, dataset):

    answer = []

    if dataset=="hh101":
        on_threshold = 10.
    elif dataset=="adlmr":
        on_threshold = 5.
    elif dataset=="testbed":
        on_threshold = 60.
    else:
        raise ValueError("Wrong dataset")

    sensor_list = sorted(set(events[:,0]))
    start_time = float(events[0,2])

    active_states = {item: [] for item in sensor_list}
    current_state = {item: False for item in sensor_list}
    active_start = {item: 0. for item in sensor_list}
    active_length = {item: 0 for item in sensor_list}

    # Initialize correlation (ALL zero? or location-based init values?)
    correlation = np.zeros((len(sensor_list), len(sensor_list)))
    processed = np.zeros((len(sensor_list), len(sensor_list)), dtype=int)

    for i in range(len(events)):
        sensor, value, time_ = events[i][:3]
        time_=float(time_)-start_time
        tindex = sensor_list.index(sensor)

        # Set
        if value in ["ON", "true" ,"OPEN"]:
            if current_state[sensor]==False: # NEW ACTIVE STATE
                active_start[sensor]=time_  # Record Active Start Time
                current_state[sensor]=True  # Current Acting
                active_length[sensor]+=1    # Length = 1

            else:                               # ON and ON
                active_length[sensor]+=1        # Length = Length += 1
                print(i, sensor, time_-active_start[sensor], "ON and ON (Continue)")      
                answer.append(correlation.copy())

                continue

        elif value in ["OFF", "false", "CLOSE"]:
            if current_state[sensor]==True: # Active End
                active_states[sensor].append( # Add new active record
                    (active_start[sensor], time_, active_length[sensor])
                )
                current_state[sensor]=False
                active_length[sensor]=0
            else:                           # OFF and OFF
                print(i, sensor, time_-active_start[sensor], "OFF and OFF (Continue)")
                answer.append(correlation.copy())
                
                continue
        else:
            print("?", sensor, value, time_)

        # Calc Current Score
        if value in ["ON", "true" ,"OPEN"]:
            
            for ck, cv in current_state.items():    # Other Sensors
                if ck==sensor: continue
                cindex = sensor_list.index(ck)
                row, col = min(tindex, cindex), max(tindex, cindex)

        elif value in ["OFF", "false", "CLOSE"]:
            
            # Update First
            for ck, cv in current_state.items():
                if ck==sensor: continue
                cindex = sensor_list.index(ck)
                row, col = min(tindex, cindex), max(tindex, cindex)
                prev_score = correlation[row, col]

                if cv==True: # B (ON) - A (OFF, current)
                    
                    # Update First
                    # 1. existing active states
                    lstate = active_states[sensor][-1]
                    a_length = lstate[1]-lstate[0]
                    from_, to_ = processed[tindex, cindex], len(active_states[ck])

                    current_score = 0.

                    for pi in range(from_, to_):
                        record = active_states[ck][pi]
                        b_length = record[1]-record[0]
                        if lstate[0]<=record[1]<=lstate[1]:
                            intersection_off = record[1]-max(record[0], lstate[0])
                            # overlap_score = intersection_off*(1./a_length+1./b_length)
                            off_overlap_score = intersection_off/(a_length+b_length)
                            current_score+=off_overlap_score
                    processed[tindex, cindex]=to_

                    # 2. ongoing active state
                    intersection_on = time_-active_start[ck]
                    on_overlap_score = intersection_on/(intersection_on+a_length)
                    current_score += on_overlap_score

                    new_score = prev_score + current_score
                    correlation[row, col] = new_score

                else: # B (OFF) - A (OFF, current)
                    
                    # Update First
                    # 1. existing active states
                    lstate = active_states[sensor][-1]
                    a_length = lstate[1]-lstate[0]
                    from_, to_ = processed[tindex, cindex], len(active_states[ck])

                    current_score = 0.

                    for pi in range(from_, to_):
                        record = active_states[ck][pi]
                        b_length = record[1]-record[0]
                        if lstate[0]<=record[1]<=lstate[1]:
                            intersection_off = record[1]-max(record[0], lstate[0])
                            off_overlap_score = intersection_off*(1./a_length+1./b_length)
                            current_score+=off_overlap_score
                    processed[tindex, cindex]=to_

                    new_score = prev_score + current_score
                    correlation[row, col] = new_score
            
            # Update 완료
            # rep_sensor에 추가
            # for ck in active_states.keys():
            #     if ck==sensor: continue
            #     cindex = sensor_list.index(ck)
            #     row, col = min(tindex, cindex), max(tindex, cindex)
            #     score = correlation[row, col]

                # ON 상태인 경우, 계산 후 판단
                #   1. "현재 센서의 OFF까지의 active_state_a"와 "다른 센서의 (Process되지 않은) active_state_b와 ON 시작부터 현재까지의 partial_active_b"를 가지고 연관도를 계산함
                #       연관도 계산법
                #           t에서의 최종 연관도 = (t-1)에서의 최종 연관도 + t에서의 현재 기준 연관도
                #           t에서의 현재 기준 연관도
                #               - active_state_a
                #               - active_state_b (>=0), partial_active_b
                #               a.  active_state_a와 active_state_b
                #               b.  active_state_a와 partial_active_b
                #           
                #   2. A의 Table에서 B에 해당하는 연관도 값
                #       a. threshold를 넘으면 A는 B로 진행된다고 볼 수 있음
                #       b. threshold를 넘지 못하면 A는 B로 진행되는 것이 아님
                #   3. 1-2 과정을 나머지 센서들에 대해 적용한다
                #   4. 결과적으로, ON 상태의 나머지 센서들 중에서 A와 연관도가 높은 센서가
                #       a. 하나 이상이면, 해당 센서 (B)에 A의 정보를 넘겨줌
                #       b. 아무것도 없으면, 

                # OFF 상태인 경우, 계산 후 판단
                #   1. "현재 센서의 OFF까지의 active_state_a"와 "다른 센서의 (Process되지 않은) active_state_b"를 가지고 연관도를 계산함
                #       연관도 계산법
                #   2. 

        else:
            pass

        for k, v in active_start.items(): # True인 상태로 Threshold를 초과했을 때 처리하기
            if current_state[k]==True and time_-v>on_threshold:
                print(i, k)
                active_states[k].append((v, v+on_threshold, active_length[k]))
                current_state[k]=False
                active_length[k]=0

        answer.append(correlation.copy())

        if i==len(events)-1:
            for k, v in current_state.items():
                if v==True:
                    active_states[k].append((active_start[k], time_, active_length[k]))
                    current_state[k]=False

    assert len(events)==len(answer)
    # return correlation
    return answer