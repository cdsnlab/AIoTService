import numpy as np
from .info.config import config
from scipy.signal import find_peaks

def episode_evaluation(scores, episode, gt, interval=10, metric='SEP'):
    """

    """

    gt_timestamp_b, gt_timestamp_a=float(episode[gt-1][2]), float(episode[gt][2])


    # if metric.lower()=='sep':
        # filter -> peak
    scores_=np.where(scores<0.3, 0, scores)
    positive_, _=find_peaks(scores_)
    # elif metric.lower()=='rulsif':

    
    #     # # peak -> filter
    #     # peak_2, _=find_peaks(scores)
    #     # positive_=[p for p in peak_2 if scores[p]>0.3]

    true_positive, false_positive, true_negative, false_negative = [], [], [], []
    target_positive =positive_

    for i, event in enumerate(episode):
        if i in target_positive: # POSITIVE
            if i==gt:
                true_positive.append(i)
            else:
                event_time=float(event[2])
                if abs(gt_timestamp_a-event_time)<interval or abs(gt_timestamp_b-event_time)<interval:
                    true_positive.append(i)
                else:
                    false_positive.append(i)
        else:   # NEGATIVE
            if i==gt:
                false_negative.append(i)
            else:
                true_negative.append(i)

    return true_positive, false_positive, true_negative, false_negative

def all_evaluation(scores, events, gt_list, order='fp'):
    """

    """

    gt_timestamp_b= np.array([float(events[idx][2]) for idx in gt_list])
    gt_timestamp_a= np.array([float(events[idx+1][2]) for idx in gt_list])
    

    if order=='fp':
        # filter -> peak
        scores_1=np.where(scores<0.3, 0, scores)
        positive_, _=find_peaks(scores_1)
    else:
        # peak -> filter
        peak_2, _=find_peaks(scores)
        positive_=[p for p in peak_2 if scores[p]>0.3]

    true_positive, false_positive, true_negative, false_negative = [], [], [], []
    target_positive =positive_

    for i, event in enumerate(events):
        if i in target_positive: # POSITIVE
            if i in gt_list:
                true_positive.append(i)
            else:
                event_time=float(event[2])
                if sum(abs(gt_timestamp_a-event_time)<15)!=0 or \
                    sum(abs(gt_timestamp_b-event_time)<15)!=0:
                    true_positive.append(i)
                else:
                    false_positive.append(i)
        else:   # NEGATIVE
            if i in gt_list:
                false_negative.append(i)
            else:
                true_negative.append(i)

    return true_positive, false_positive, true_negative, false_negative


def evaluation_adlmr(scores, events, gts):
    threshold, interval=config['threshold'], config['interval']
    fp,fn,tp,tn=[],[],[],[]

    # gt1, gt2=float(events[gt][-1]), float(events[gt-1][-1])
    gts1=np.array([float(events[idx][2]) for idx in gts])#; gts2=np.array([float(events[idx-1][2]) for idx in gts])
    for i, event in enumerate(events):
        score=scores[i]
        if score>threshold: # positive
            timestamp=float(events[i][2])
            if sum(abs(gts1-timestamp)<interval):# or sum(abs(gts2-timestamp)<interval):
                tp.append(i)
            else:
                fp.append(i)
        else:
            if i in gts:
                fn.append(i)
            else:
                tn.append(i)
    
    return fp, fn, tp, tn

