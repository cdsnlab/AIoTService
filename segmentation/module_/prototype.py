import numpy as np
from itertools import combinations
def coefficient(events, lambda_):

    sensor_list = sorted(set(events[:,0]))

    coefficient_matrix = np.diag([0. for _ in sensor_list])

    keys = list(combinations([i for i in range(len(sensor_list))], 2))
    coef_records = {
        "{}{}".format(min(l,r),max(l,r)):[] for l, r in keys
    }

    assert len(coef_records.keys())==len(keys)


    states = []
    for ei in range(len(events)):
        sensor_i, timestamp = events[ei, 0], float(events[ei, 2])
        i_idx = sensor_list.index(sensor_i)

        # create window
        window = events[max(0, ei-lambda_):ei+lambda_+1] # lambda_+1 <= len(window) <= 2*lambda_+1
        # duration = float(window[-1, 2])-float(window[0, 2])

        wcount = {sensor:0 for sensor in sensor_list}
        wt = {sensor:[] for sensor in sensor_list}

        for wi in range(len(window)):
            sensor_j = window[wi, 0]
            wcount[sensor_j]+=1
            # if sensor_i == sensor_j:
            #     continue
            wt[sensor_j].append(float(window[wi, 2]))

        assert sum(wcount.values())==len(window)

        denominator = float(len(window)-wcount[sensor_i])
        for j, (sensor_j, v) in enumerate(wt.items()):
            if sensor_i == sensor_j or len(v) == 0: # Itself OR No sensor in window
                continue
            coef_ij = sum([1./(1.+abs(timestamp-t)) for t in v]) * (wcount[sensor_j]/denominator) # Weighted SUM

            key = "{}{}".format(min(i_idx, j), max(i_idx, j))
            coef_records[key].append((timestamp, coef_ij))

            accum_coef_ij = 0
            # if len(coef_records[key])>=lambda_:
            for pt, pc in coef_records[key][-lambda_ if len(coef_records[key])>=lambda_ else 0:]:
                accum_coef_ij+=pc/(1.+timestamp-pt)

            # print(sensor_i, sensor_j, coef_records[key], accum_coef_ij)
            # else:
            #     for pt, pc in coef_records[key]:
            #         accum_coef_ij+=pc/(1.+timestamp-pt)

            coefficient_matrix[i_idx, j] = accum_coef_ij
            coefficient_matrix[j, i_idx] = accum_coef_ij

        # How can we treat sensors not included in the window..
        

        states.append(np.mean(
            coefficient_matrix[i_idx:]
        ))

    assert len(states)==len(events)

    return np.array(states, dtype=np.float64)







    

