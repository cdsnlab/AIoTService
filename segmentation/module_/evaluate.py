import numpy as np

from .featureExtraction import feature_extraction
from .changePointDetection import change_point_detection

def cpd_calculate(pair_activities, dataset, sensors, algorithm, save=False):

    scores_list = []

    for key in sorted(pair_activities.keys()):
        for pair in pair_activities[key]:

            _, episode = pair[-2:]

            features = np.array(feature_extraction(episode, dataset, sensors))

            scores = np.array(change_point_detection(features, algorithm))
            scores[scores<0] = 0
            
            scores_list.append(scores)
    
    if save:
        np.save(f"./evaluation/{dataset}/{algorithm}/scores_{algorithm}.npy", scores_list)

    return scores_list


def cpd_evaluate(pair_activities, scores_list, time_interval, algorithm, strict):

    print(algorithm)
    if strict:
        print("Exact CPD")
    else:
        print("CPD Within 1 window")

    for threshold in range(50, 0, -5):
        # threshold /= 20.
        tp_rates, fp_rates = [], []
        score_idx = 0

        for key in sorted(pair_activities.keys()):
            for pair in pair_activities[key]:
                transition, episode = pair[-2:]
                scores = scores_list[score_idx]

                assert len(episode) == len(scores)
                assert episode[transition-1, 2] == episode[transition, 2]

                stime, etime = int(float(episode[0, 2])), int(float(episode[-1, 2]))
                ttime = int(float(episode[transition, 2]))
                duration = etime - stime

                window_length = int(duration/time_interval) + 1
                tw_idx = int((ttime-stime)/time_interval)
                change_point_window = np.zeros(window_length)

                tp = tn = fp = fn = 0

                for e in range(len(episode)):
                    s, v, t = episode[e][:3]
                    t = int(float(t)) - stime

                    e_idx = int(t/time_interval)

                    if scores[e] > threshold:
                        change_point_window[e_idx] = 1

                for ci in range(window_length):
                    if tw_idx - 1 <= ci <= tw_idx + 1:
                        if change_point_window[ci] != 0:
                            if strict:
                                if ci == tw_idx:
                                    tp += 1
                                else:
                                    fp += 1
                            else:
                                tp += 1
                        else:
                            if ci == tw_idx:
                                fn += 1
                            else:
                                tn += 1
                    else:
                        if change_point_window[ci] != 0:
                            fp += 1
                        else:
                            tn += 1

                tp_rates.append(tp/(tp+fn))
                fp_rates.append(fp/(fp+tn))

                score_idx += 1

        
        print(threshold, sum(tp_rates)/len(tp_rates), sum(fp_rates)/len(fp_rates))

