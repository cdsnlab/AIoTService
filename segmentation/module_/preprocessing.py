import glob
import collections
from itertools import combinations, permutations
import numpy as np

from .readText import read_adlmr, time_correction

def generate_pair():
    path = "./dataset/testbed/npy/seminar/MO/B"

    sensors = set()
    activities = collections.defaultdict(list)

    for file_ in sorted(glob.glob(f"{path}/*.npy")):

        filename = file_.split("/")[-1].split(".")[0]
        label, num = filename[:-2], filename[-2:]
        npy_data = np.load(file_)

        motion_data = np.array([data for data in npy_data if data[0][0]=="M"])
        sensors |= set(motion_data[:,0])

        activities[label].append((num, motion_data))

    pair_list = list(permutations(list(activities.keys()), 2)) + [(k, k) for k in activities.keys()]

    pair_activities = collections.defaultdict(list)

    for left, right in pair_list:

        for left_num, left_data in activities[left]:

            for right_num, right_data in activities[right]:
                
                pair = time_correction(np.concatenate((left_data, right_data)), len(left_data))

                pair_activities[f"{left}-{right}"].append((left_num, right_num, len(left_data), pair))

    return pair_activities, sorted(sensors)


def generate_pair_adlmr():

    limits = 10

    with open("./dataset/adlmr/annotated", "rb") as f:
        events = read_adlmr(f.readlines())

    sensors = sorted([item for item in set(np.array(events)[:,0]) if item[0]=="M"])
    activities = collections.defaultdict(list)

    curr_label = events[0][-1]
    activity = []
    for e in range(len(events)): # [s, v, t, r, l]
        if events[e][-1] == curr_label:
            activity.append(events[e])
        else:
            sample = np.array(activity.copy())
            assert len(set(sample[:,-1]))==1

            activities[curr_label].append(sample)

            activity = [events[e]]
            curr_label = events[e][-1]
        
        if e==len(events)-1:
            sample = np.array(activity.copy())
            assert len(set(sample[:,-1]))==1

            activities[curr_label].append(sample)

    pped_activities = []
    for data in activities["GF"]:
        pp_activity = []
        for row in data:
            _, _, _, r, _ = row
            if int(r) != 1:
                pp_activity.append(row)
        pped_activities.append(np.array(pp_activity))
    
    activities["GF"] = pped_activities

    # min_sample_num = min([len(v) for v in activities.values()])
    min_sample_num = limits
    for k in activities.keys():
        activities[k]=activities[k][:min_sample_num]



    pair_list = list(permutations(list(activities.keys()), 2)) + [(k, k) for k in activities.keys()]

    pair_activities = collections.defaultdict(list)

    for left, right in pair_list:

        for left_data in activities[left]:

            for right_data in activities[right]:

                pair = time_correction(np.concatenate((left_data, right_data)), len(left_data))

                pair_activities[f"{left}-{right}"].append((len(left_data), pair))

    return pair_activities, sensors



    



    

    

