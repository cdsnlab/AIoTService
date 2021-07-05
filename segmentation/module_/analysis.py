import numpy as np
import os
import matplotlib.pyplot as plt
from .info.config import config, feature_name

def neighbor_events(episodes, idx):
    left, right=idx-config['ws']-1, idx+3
    if left<0:
        target=episodes[:right]
        target_=np.array([episodes[0] for i in range(33-len(target))])
        target__=np.concatenate((target_, target), axis=0)
    else:
        target__=episodes[left:right]
    
    return target__

# def feature_combination(features):
#     for fs in features:
#         cfs=set()
#         fs_=np.round(fs, 3)
#         N=int((fs_.shape[1]-8)/2)
#         for col in range(fs_.shape[1]):
#             colv=fs_[:,col]
#             if min(colv)!=max(colv):
#                 if col<8:
#                     cfs.add(col)
#                 else:
#                     if col>=8 and col<8+N:
#                         cfs.add(8)
#                     else:
#                         cfs.add(9)
#         stl=sorted(list(cfs))
#         stl=[str(item) for item in stl]

#     return "".join(stl)

def single_feature(tag, features, test_gt, test, sensor_list): # TODO

    scope=50
    # print(test_gt-scope, test_gt+scope)

    t_lb, t_ub = max(0, test_gt-scope), min(test_gt+scope, len(test))
    transition_nearby=features[t_lb:t_ub]

    l_lb, l_ub = max(0, int(test_gt/2)-scope), min(int(test_gt/2)+scope, test_gt)
    non_range_l=features[l_lb:l_ub]

    r_lb, r_ub = max(test_gt, int(test_gt+(len(test)-test_gt)/2)-scope), min(int(test_gt+(len(test)-test_gt)/2)+scope, len(test))
    non_range_r=features[r_lb:r_ub]


    # target_features=transition_nearby # *_*
    target_features=non_range_l
    target_indices=range(l_lb, l_ub) # *_*

    print(target_features.shape)
    # target_scores=scores[t_lb:t_ub]
    # plt.axhline(y=0.3, linestyle=":", color='c')
    # plt.plot(target_indices, target_scores, label="score")

    fixed_ranges=range(0, 12)
    count_ranges=range(12, 12+len(sensor_list))
    elapsed_ranges=range(12+len(sensor_list), target_features.shape[1])

    # print(sensor_list)
    plt.ylim(0, 1.5)
    # plt.margins(1, 1)
    for colnum in elapsed_ranges:
        col=target_features[:,colnum]
        sid=-1

        if colnum<12:
            pass
        elif colnum>=12 and colnum<12+len(sensor_list):
            sid=colnum-12
            colnum=12
        else:
            sid=colnum-12-len(sensor_list)
            colnum=13
        
        if sid!=-1:
            label="{} {}".format(feature_name[colnum], sensor_list[sid])
        else:
            label=feature_name[colnum]

        # if colnum<3: # time features
        # if colnum==3: # window duration
        # if colnum==4: # time since last sensor event
        # if colnum==5 or colnum==6: # dominant sensors from previous windows
        # if colnum==7 or colnum==8: # last sensor id and location
        # if colnum==9: # last motion location
        # if colnum==10: # entropy
        # if colnum==11: # activity level change
            # continue
            # if max(col)!=min(col):
            # print(col)
        plt.plot(target_indices, col, label=label)
            # ncol+=1
    
    feat_name='elapsed' # *_*
    if not os.path.exists("analysis/testbed/{}".format(tag)):
        os.mkdir("analysis/testbed/{}".format(tag))
    if not os.path.exists("analysis/testbed/{}/{}".format(tag, feat_name)):
        os.mkdir("analysis/testbed/{}/{}".format(tag, feat_name))
    
    filename="analysis/testbed/{}/{}/t_{}.png".format(tag, feat_name, test_gt)
    if filename.split("/")[-1][0]=='t':
        plt.axvline(x=test_gt, linestyle=':', color='r', label='transition')

    plt.savefig(filename)
    plt.clf()

def false_alarms(): # TODO

    # part_scores=np.array(scores)
    # part_scores[part_scores<0.3]=0
    # part_events=test

    # if test_gt-30<0 or test_gt+30>=len(test):
    #     continue
    # range_transition_nearby=range(test_gt-30, test_gt+30)

    # alarms=0
    # for i in range_transition_nearby:
    #     if part_scores[i]>0.45:
    #         alarms+=1

    # label=tag[0]+tag[2]

    # if label not in transition_occurrences.keys():
    #     transition_occurrences[label]={"alarms":0, "combinations":0}
    #     transition_occurrences[label]['alarms']+=alarms
    #     transition_occurrences[label]['combinations']+=1
    # else:
    #     transition_occurrences[label]['alarms']+=alarms
    #     transition_occurrences[label]['combinations']+=1

    # plt.plot(range(len(part_scores)), part_scores, marker='o')
    # plt.title(tag)

    # plt.ylim(-0.1, 1.1)

    # plt.axvline(x=test_gt, linestyle=":", color='g')
    # plt.axvspan(xmin=0, xmax=30, alpha=0.2)
    
    # filename="analysis/testbed/graph/{}.png".format(tag)

    # plt.savefig(filename)
    # plt.clf()
    return

def false_alarm(): # TODO
    # non_peak=[i for i in range(len(scores)) if scores[i]>0.3]
    # peak, _=find_peaks(scores)
    # peak=[p for p in peak if scores[p]>0.3]
    # plt.title(tag)
    # plt.xlabel('events'); plt.ylabel('score')
    # plt.plot(range(len(scores)), scores, 'ro-')
    # plt.plot(peak, scores[peak], '*')
    # plt.axvline(x=test_gt, linestyle=':')

    # ## peak-check
    # bucket={item:0 for item in ['tp','fp','tn','fn']}
    # gtb, gta=float(test[test_gt-1][2]), float(test[test_gt][2]) # time-wise
    # # gtb, gta=test_gt-1, test_gt
    # for i, event in enumerate(test):
    #     if i in peak: # positive
    #         if i==test_gt:
    #             bucket['tp']+=1
    #         else:
    #             it=float(test[i][2])
    #             if abs(it-gtb)<10 or abs(it-gta)<10:
    #             # if abs(i-gtb)<5 or abs(i-gta)<5:
    #                 bucket['tp']+=1
    #             else:
    #                 bucket['fp']+=1
    #     else: # negative
    #         if i==test_gt:
    #             bucket['fn']+=1
    #         else:
    #             bucket['tn']+=1

    # fp_+=bucket['fp']
    # print(bucket, fp_/(e_num+1))

    return