from numpy import dtype
import re
import time
import glob
import pickle
import math
import datetime
from abc import *
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Dataloader(Sequence):
    def __init__(self, indices, x_set, y_set, len_set, count_set, batch_size, shuffle=False, tr_points=None, tr_boundary=None, window_ratio=0.1, aug_multiple=0):
    # def __init__(self, indices, x_set, y_set, len_set, count_set, batch_size, prev_y_set, shuffle=False, tr_points=None, tr_boundary=None):
        self.indices = indices
        self.x, self.y, self.len, self.count = x_set, y_set, len_set, count_set
        # self.prev_y = prev_y_set
        self.tr_points, self.tr_boundary = tr_points, tr_boundary
        if aug_multiple > 0:
            self.x, self.y, self.len, self.count = self.x[indices], self.y[indices], self.len[indices], self.count[indices]
            self.augmentation(window_ratio, aug_multiple)
            self.indices = np.array(range(self.x.shape[0]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        print(f'The number of batches: {math.ceil(len(self.indices) / self.batch_size)}')

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_x = np.array([self.x[i] for i in indices])
        batch_y = np.array([self.y[i] for i in indices])
        batch_len = np.array([self.len[i] for i in indices])
        batch_count = np.array([self.count[i] for i in indices])
        batch_tr_point = np.array([self.tr_points[i] for i in indices])
        # batch_prev_y = np.array([self.prev_y[i] for i in indices])
        # if self.tr_points is not None and self.tr_boundary is not None:
        #     batch_tr_point = np.array([self.tr_points[i] for i in indices])
        #     batch_tr_boundary = np.array([self.tr_boundary[i] for i in indices])
        # else:
        #     batch_tr_point = None
        #     batch_tr_boundary = None
        return batch_x, batch_y, batch_len, batch_count, batch_tr_point
        # return batch_x, batch_y, batch_len, batch_count, batch_prev_y
        
    def augmentation(self, window_ratio, multiple):
        X, Y, lengths, event_counts, noise_amount = [], [], [], [], []
        X.append(self.x)
        Y += list(self.y)
        lengths += list(self.len)
        event_counts += list(self.count)
        noise_amount += list(self.tr_points)
        for x, y, l, e, n in zip(self.x, self.y, self.len, self.count, self.tr_points):
            X.append(self.window_warp(x, l, window_ratio=window_ratio, multiple=multiple))
            Y += [y] * multiple
            lengths += [l] * multiple
            event_counts += [e] * multiple
            noise_amount += [n] * multiple
        self.x = np.concatenate(X)
        self.y = np.array(Y)
        self.len = np.array(lengths)
        self.count = np.array(event_counts)
        self.tr_points = np.array(noise_amount)
    
    def window_warp(self, x, length, window_ratio=0.1, scales=[0.5, 2.], multiple=3):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(scales, multiple)
        warp_size = np.ceil(window_ratio*length).astype(int)
        window_steps = np.arange(warp_size)
        
        window_starts = np.random.randint(low=1, high=length-warp_size-1, size=multiple).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
        
        time_steps, channel = x.shape        
        ret = np.zeros([multiple, time_steps, channel])
        for i in range(multiple):
            for dim in range(channel):
                start_seg = x[:window_starts[i],dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, x[window_starts[i]:window_ends[i],dim])
                end_seg = x[window_ends[i]:length,dim]
                padding_seg = x[length:,dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))                
                warped = np.interp(np.arange(length), np.linspace(0, length-1., num=warped.size), warped).T
                ret[i,:,dim] = np.concatenate((warped, padding_seg))
        return ret

    def on_epoch_end(self):
        # self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# class AmbientData(metaclass=ABCMeta):
#     def __init__(self, args):
#         # self.seq_len = args.seq_len
#         self.idx2label = {}
#         self.label2idx = {}
#         self.N_FEATURES = len(self.sensors)
#         self.sensor2index = {sensor: i for i, sensor in enumerate(self.sensors)}
#         self.X, self.Y, self.lengths, self.event_counts = self.generateDataset()
#         self.nseries, _, _ = self.X.shape
#         self.N_CLASSES = len(np.unique(self.Y))

#     def change_state(self, activated, s, v):
#         vector = activated.copy()
#         if v.lower() in ["on", "true"]:
#             vector[self.sensor2index[s]] = 1
#         else:
#             vector[self.sensor2index[s]] = 0
#         return vector

#     def event2matrix(self, episode):
#         activated = np.zeros(self.N_FEATURES)
#         start_time = int(float(episode[0][2]))
#         episode[:,2] = list(map(lambda x: int(float(x) - start_time), episode[:,2]))
#         duration = int(episode[-1][2])
#         if duration < 1:
#             return None
#         # state_matrix = np.zeros((int(episode[-1][2]) + 1, self.N_FEATURES))
#         state_matrix = np.zeros((self.args.seq_len, self.N_FEATURES))
#         prev_t = 0
#         count = 0
#         count_seq = np.zeros((self.args.seq_len))
#         for s, v, t, l in episode:
#             t = int(t)
#             if t >= self.args.seq_len:  # Only for the episode whose sequence length exceeds the predefined seq_len
#                 t = self.args.seq_len - 1
#                 state_matrix[prev_t:t] = activated
#                 count_seq[prev_t:t] = count
#                 activated = self.change_state(activated, s, v)
#                 break
#             if t != prev_t:
#                 state_matrix[prev_t:t] = activated
#                 count_seq[prev_t:t] = count
#                 prev_t = t
#             activated = self.change_state(activated, s, v)
#             count += 1
#         state_matrix[t] = activated
#         count_seq[t] = count
#         return [np.array(state_matrix), l, duration+1, count_seq, t, start_time]

#     def generateDataset(self):
#         # if self.args.rnd_prefix:
#         #     self.sample_suffix()
#         X, Y, lengths, event_counts, start_time = [], [], [], [], []
#         for episode in self.episodes:
#             # if self.args.remove_prefix:
#             #     episode = episode[self.args.prefix_len:, :] if len(episode) > self.args.prefix_len else episode
#             converted = self.event2matrix(episode)
#             if converted is None:
#                 continue
#             # if self.args.rnd_prefix:
#             #     idx = np.random.choice(len(self.suffix), 1)[0]
#             #     converted[0] = np.concatenate((self.suffix[idx], converted[0]), axis=0)
#             if self.args.expiration_period != -1:
#                 X.append(self.activation_expire(converted[0]))
#             else:
#                 X.append(converted[0])
#             Y.append(converted[1])
#             lengths.append(converted[2])
#             event_counts.append(converted[3])
#             start_time.append(converted[5])
#         X = pad_sequences(X, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len)  # B * T * V
#         self.idx2label = {i:label for i, label in enumerate(sorted(set(Y)))}
#         self.label2idx = {label:i for i, label in self.idx2label.items()}
#         Y = [self.label2idx[l] for l in Y]
#         Y = np.array(Y)
#         lengths = np.array(lengths)
#         event_counts = np.array(event_counts)
#         self.start_time = np.array(start_time)
#         return X, Y, lengths, event_counts
    
#     def activation_expire(self, state_matrix):
#         count = np.zeros((self.N_FEATURES))
#         flag = np.zeros((self.N_FEATURES))
#         temp_state_matrix = []
#         for state in state_matrix:
#             new_ON_idx = np.where((state==1) & (flag == 0))[0]
#             new_OFF_idx = np.where(((state==0) & (flag == 1)) | (count>=self.args.expiration_period))[0]
            
#             flag[new_ON_idx] = 1
#             flag[new_OFF_idx] = 0
#             count[new_OFF_idx] = 0
            
#             activated_idx = np.where(flag==1)[0]
#             count[activated_idx] += 1
#             temp_state_matrix.append(flag.copy())
#         state_matrix = np.concatenate(temp_state_matrix).reshape((-1, self.N_FEATURES))
#         return state_matrix


# class CASAS_ADLMR(AmbientData):
#     def __init__(self, args):
#         self.args = args
#         self.main()
#         super().__init__(args)
    
#     def main(self):
#         self.filename = './dataset/adlmr_collaborative'
#         with open(self.filename, 'rb') as f:
#             self.adlmr = pickle.load(f)
#         self.episodes = self.adlmr['episodes']
#         self.sensors = self.adlmr['sensors']
        
#         col = [0, 1, 2, 4]
#         episodes = []
#         for episode in self.episodes:
#             episodes.append(episode[:, col])
#         self.episodes = episodes
  

# args.with_other=False
# args.random_noise=True
# args.except_all_other_events=True
# data = CASAS_RAW_NATURAL(args)
# sensors, values, timestamps, activities = data.preprocessing()

# temp_sensors, temp_values, temp_timestamps, temp_activities = [], [], [], []
# other_t = []
# total_offset = 0
# for s, v, t, l in zip(sensors, values, timestamps, activities):
#     if data.mappingActivities[data.args.dataset][l] == 'Other':
#         other_t.append(t)
#         prev_l = l
#         continue
#     if data.mappingActivities[data.args.dataset][prev_l] == 'Other':
#         total_offset += other_t[-1] - other_t[0]
#         other_t = []
#     temp_sensors.append(s)
#     temp_values.append(v)
#     temp_timestamps.append(t - total_offset)
#     temp_activities.append(l)
#     prev_l = l

# prev_t = 0
# count = 0
# for i, t in enumerate(temp_timestamps):
#     if prev_t > t:
#         count += 1
#     prev_t = t
    


class CASAS_RAW_SEGMENTED:
    def __init__(self, args):
        self.args = args
        self.main()
        
    def main(self):
        self.mappingActivities = {
                    "cairo": {"": "Other",
                              "R1wake": "Other",
                              "R2wake": "Other",
                              "Nightwandering": "Other",
                              "R1workinoffice": "Work",
                              "Laundry": "Work",
                              "R2takemedicine": "Take_medicine",
                              "R1sleep": "Sleep",
                              "R2sleep": "Sleep",
                              "Leavehome": "Leave_Home",
                              "Breakfast": "Eat",
                              "Dinner": "Eat",
                              "Lunch": "Eat",
                              "Bedtotoilet": "Bed_to_toilet"},
                    "kyoto7": {"R1_Bed_to_Toilet": "Bed_to_toilet",
                               "R2_Bed_to_Toilet": "Bed_to_toilet",
                               "Meal_Preparation": "Cook",
                               "R1_Personal_Hygiene": "Personal_hygiene",
                               "R2_Personal_Hygiene": "Personal_hygiene",
                               "Watch_TV": "Relax",
                               "R1_Sleep": "Sleep",
                               "R2_Sleep": "Sleep",
                               "Clean": "Work",
                               "R1_Work": "Work",
                               "R2_Work": "Work",
                               "Study": "Other",
                               "Wash_Bathtub": "Other",
                               "": "Other"},
                    "kyoto8": {"R1_shower": "Bathing",
                               "R2_shower": "Bathing",
                               "Bed_toilet_transition": "Other",
                               "Cooking": "Cook",
                               "R1_sleep": "Sleep",
                               "R2_sleep": "Sleep",
                               "Cleaning": "Work",
                               "R1_work": "Work",
                               "R2_work": "Work",
                               "": "Other",
                               "Grooming": "Other",
                               "R1_wakeup": "Other",
                               "R2_wakeup": "Other"},
                    "kyoto11": {"": "Other",
                                "R1_Wandering_in_room": "Other",
                                "R2_Wandering_in_room": "Other",
                                "R1_Work": "Work",
                                "R2_Work": "Work",
                                "R1_Housekeeping": "Work",
                                "R1_Sleeping_Not_in_Bed": "Sleep",
                                "R2_Sleeping_Not_in_Bed": "Sleep",
                                "R1_Sleep": "Sleep",
                                "R2_Sleep": "Sleep",
                                "R1_Watch_TV": "Relax",
                                "R2_Watch_TV": "Relax",
                                "R1_Personal_Hygiene": "Personal_hygiene",
                                "R2_Personal_Hygiene": "Personal_hygiene",
                                "R1_Leave_Home": "Leave_Home",
                                "R2_Leave_Home": "Leave_Home",
                                "R1_Enter_Home": "Enter_home",
                                "R2_Enter_Home": "Enter_home",
                                "R1_Eating": "Eat",
                                "R2_Eating": "Eat",
                                "R1_Meal_Preparation": "Cook",
                                "R2_Meal_Preparation": "Cook",
                                "R1_Bed_Toilet_Transition": "Bed_to_toilet",
                                "R2_Bed_Toilet_Transition": "Bed_to_toilet",
                                "R1_Bathing": "Bathing",
                                "R2_Bathing": "Bathing"},
                    "milan": {"": "Other",
                              "Master_Bedroom_Activity": "Other",
                              "Meditate": "Other",
                              "Chores": "Work",
                              "Desk_Activity": "Work",
                              "Morning_Meds": "Take_medicine",
                              "Eve_Meds": "Take_medicine",
                              "Sleep": "Sleep",
                              "Read": "Relax",
                              "Watch_TV": "Relax",
                              "Leave_Home": "Leave_Home",
                              "Dining_Rm_Activity": "Eat",
                              "Kitchen_Activity": "Cook",
                              "Bed_to_Toilet": "Bed_to_toilet",
                              "Master_Bathroom": "Bathing",
                              "Guest_Bathroom": "Bathing"}}
        self.filename = f'./dataset/{self.args.dataset}'
        sensors, values, timestamps, activities = self.preprocessing()
        if self.args.except_all_other_events:
            sensors, values, timestamps, activities = self.exclude_other_events(sensors, values, timestamps, activities)            
        self.sensors = sorted(set(sensors))
        self.N_FEATURES = len(self.sensors)
        self.sensor2index = {sensor: i for i, sensor in enumerate(self.sensors)}
        self.episodes = self.create_episodes(sensors, values, timestamps, activities)
        if self.episodes is not None:
            self.X, self.Y, self.lengths, self.event_counts = self.generateDataset()
        self.nseries, _, _ = self.X.shape
        self.N_CLASSES = len(np.unique(self.Y))
    
    def preprocessing(self):
        activity = ''  # empty
        sensors, values, timestamps, activities = [], [], [], []
        with open(self.filename, 'rb') as features:
            database = features.readlines()
            for i, line in enumerate(database):  # each line
                f_info = line.decode().split()  # find fields
                try:
                    # if 'M' == f_info[2][0] or 'D' == f_info[2][0]:
                    # choose only M D sensors, avoiding unexpected errors
                    if '.' not in f_info[1]:
                        f_info[1] = f_info[1] + '.000000'
                    s = str(f_info[0]) + str(f_info[1])
                    timestamps.append(int(time.mktime(datetime.strptime(s, "%Y-%m-%d%H:%M:%S.%f").timetuple())))
                    if f_info[3] == 'OPEN':
                        f_info[3] = 'ON'
                    elif f_info[3] == 'CLOSE':
                        f_info[3] = 'OFF'
                    sensors.append(f_info[2])
                    values.append(f_info[3])

                    if len(f_info) == 4:  # if activity does not exist
                        activities.append(activity)
                    else:  # if activity exists
                        des = ''.join(f_info[4:])
                        if 'begin' in des:
                            activity = re.sub('begin', '', des)
                            activities.append(activity)
                        # if 'end' in des and activity == re.sub('end', '', des):
                        if 'end' in des:
                            activities.append(activity)
                            activity = ''
                    if f_info[2][0] not in ['M', 'D']:
                        del sensors[-1]
                        del values[-1]
                        del timestamps[-1]
                        del activities[-1]
                except IndexError:
                    print(i, line)
        features.close()
        # sensors, values, timestamps, activities = self.sort_by_time(sensors, values, timestamps, activities)
        return sensors, values, timestamps, activities
    
    def create_episodes(self, sensors, values, timestamps, activities):
        X, x = [], []
        # Y = []
        prev_label = None
        for s, v, t, l in zip(sensors, values, timestamps, activities):
            if prev_label == l or prev_label is None:
                x.append([s, v, t, self.mappingActivities[self.args.dataset][l]])
            else:
                X.append(np.array(x))
                # Y.append(mappingActivities[prev_label])
                x = [[s, v, t, self.mappingActivities[self.args.dataset][l]]]
            prev_label = l
        X.append(np.array(x))
        # Y.append(mappingActivities[prev_label])
        return X

    # def change_state(self, activated, s, v):
    #     vector = activated.copy()
    #     if v.lower() in ["on", "true", "up"]:
    #         vector[self.sensor2index[s]] = 1
    #     else:
    #         vector[self.sensor2index[s]] = 0
    #     return vector
    
    def change_state(self, activated, s, v):
        vector = activated.copy()
        if v.lower() in ["on", "true", "up"]:
            vector[self.sensor2index[s]] = 1
        elif v.lower() in ["off", "false", "down"]:
            vector[self.sensor2index[s]] = 0
        else:
            vector[self.sensor2index[s]] = float(v)
        return vector

    def event2matrix(self, episode):
        activated = np.zeros(self.N_FEATURES)
        start_time = int(float(episode[0][2]))
        episode[:,2] = list(map(lambda x: int(float(x) - start_time), episode[:,2]))
        duration = int(episode[-1][2])
        if duration < 1:
            return None
        # state_matrix = np.zeros((int(episode[-1][2]) + 1, self.N_FEATURES))
        state_matrix = np.zeros((self.args.seq_len, self.N_FEATURES))
        # if self.args.dataset == 'lapras':
        #     state_matrix -= 1
        prev_t = 0
        count = 0
        count_seq = np.zeros((self.args.seq_len))
        for s, v, t, l in episode:
            t = int(t)
            if t >= self.args.seq_len:  # Only for the episode whose sequence length exceeds the predefined seq_len
                t = self.args.seq_len - 1
                state_matrix[prev_t:t] = activated
                count_seq[prev_t:t] = count
                activated = self.change_state(activated, s, v)
                break
            if t != prev_t:
                state_matrix[prev_t:t] = activated
                count_seq[prev_t:t] = count
                prev_t = t
            activated = self.change_state(activated, s, v)
            count += 1
        state_matrix[t] = activated
        count_seq[t] = count
        return [np.array(state_matrix), l, duration+1, count_seq, t, start_time]

    def generateDataset(self):
        # if self.args.rnd_prefix:
        #     self.sample_suffix()
        X, Y, lengths, event_counts, start_time = [], [], [], [], []
        for episode in self.episodes:
            # if self.args.remove_prefix:
            #     episode = episode[self.args.prefix_len:, :] if len(episode) > self.args.prefix_len else episode
            converted = self.event2matrix(episode)
            if converted is None:
                continue
            # if self.args.rnd_prefix:
            #     idx = np.random.choice(len(self.suffix), 1)[0]
            #     converted[0] = np.concatenate((self.suffix[idx], converted[0]), axis=0)
            if self.args.expiration_period != -1:
                X.append(self.activation_expire(converted[0]))
            else:
                X.append(converted[0])
            Y.append(converted[1])
            lengths.append(converted[2])
            event_counts.append(converted[3])
            start_time.append(converted[5])
        X = pad_sequences(X, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len)  # B * T * V
        self.idx2label = {i:label for i, label in enumerate(sorted(set(Y)))}
        self.label2idx = {label:i for i, label in self.idx2label.items()}
        Y = [self.label2idx[l] for l in Y]
        Y = np.array(Y)
        lengths = np.array(lengths)
        event_counts = pad_sequences(event_counts, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len, value=0.0)
        # event_counts = np.array(event_counts)
        self.start_time = np.array(start_time)
        return X, Y, lengths, event_counts
    
    def activation_expire(self, state_matrix):
        count = np.zeros((self.N_FEATURES))
        flag = np.zeros((self.N_FEATURES))
        temp_state_matrix = []
        for state in state_matrix:
            # if self.args.dataset == 'lapras' and np.sum(state) == -self.N_FEATURES:
            #     temp_state_matrix.append(padding.copy())
            #     continue
            new_ON_idx = np.where((state==1) & (flag == 0))[0]
            new_OFF_idx = np.where(((state==0) & (flag == 1)) | (count>=self.args.expiration_period))[0]
            
            flag[new_ON_idx] = 1
            flag[new_OFF_idx] = 0
            count[new_OFF_idx] = 0
            
            activated_idx = np.where(flag==1)[0]
            count[activated_idx] += 1
            temp_state_matrix.append(flag.copy())
        state_matrix = np.concatenate(temp_state_matrix).reshape((-1, self.N_FEATURES))
        return state_matrix
    
    def sort_by_time(self, sensors, values, timestamps, activities):
        df = pd.DataFrame({'sensors': sensors, 'values': values,
                        'timestamps': timestamps, 'activities': activities})
        df.sort_values(by=['timestamps'], inplace=True)
        return df['sensors'].tolist(), df['values'].tolist(), df['timestamps'].tolist(), df['activities'].tolist()
    
    def exclude_other_events(self, sensors, values, timestamps, activities):
        temp_sensors, temp_values, temp_timestamps, temp_activities = [], [], [], []
        other_t = []
        total_offset = 0
        for s, v, t, l in zip(sensors, values, timestamps, activities):
            if self.mappingActivities[self.args.dataset][l] == 'Other':
                other_t.append(t)
                prev_l = l
                continue
            if self.mappingActivities[self.args.dataset][prev_l] == 'Other':
                total_offset += other_t[-1] - other_t[0]
                other_t = []
            temp_sensors.append(s)
            temp_values.append(v)
            temp_timestamps.append(t - total_offset)
            temp_activities.append(l)
            prev_l = l
        print("Events for Other class are excluded.")
        return temp_sensors, temp_values, temp_timestamps, temp_activities
    

class CASAS_RAW_NATURAL(CASAS_RAW_SEGMENTED):
    def __init__(self, args):
        self.args = args
        self.main()
        # self.nseries, _, _ = self.X.shape
        # self.N_CLASSES = len(np.unique(self.Y))
    
    def activation_expire(self):
        count = np.zeros((self.N_FEATURES))
        flag = np.zeros((self.N_FEATURES))
        temp_state_matrix = []
        for state in self.state_matrix:
            new_ON_idx = np.where((state==1) & (flag == 0))[0]
            new_OFF_idx = np.where(((state==0) & (flag == 1)) | (count>=10))[0]
            
            flag[new_ON_idx] = 1
            flag[new_OFF_idx] = 0
            count[new_OFF_idx] = 0
            
            activated_idx = np.where(flag==1)[0]
            count[activated_idx] += 1
            temp_state_matrix.append(flag.copy())
        self.state_matrix = np.concatenate(temp_state_matrix).reshape((-1, self.N_FEATURES))
        print("limit the duration of the sensor activation")
    
    def create_episodes(self, sensors, values, timestamps, activities):
        self.state_matrix, self.labels, prev_counts = self.event2matrix(sensors, values, timestamps, activities)
        if self.args.except_all_other_events:
            self.activation_expire()
        prev_count = 0
        prev_label = None
        x, counts = [], []
        X, Y, lengths, event_counts  = [], [], [], []
        org_Y = []
        for s, l, c in zip(self.state_matrix, self.labels, prev_counts):
            if prev_label == l or prev_label is None:
                x.append(s)
                counts.append(c)
            else:
                # if len(x) > 2:
                X.append(np.array(x))
                Y.append(self.mappingActivities[self.args.dataset][prev_label])
                org_Y.append(prev_label)
                lengths.append(X[-1].shape[0])
                event_counts.append(np.array(counts) - prev_count)
                prev_count = counts[-1]
                x = [s]
                counts = [c]
            prev_label = l
        X.append(np.array(x))
        Y.append(self.mappingActivities[self.args.dataset][prev_label])
        org_Y.append(prev_label)
        lengths.append(X[-1].shape[0])
        event_counts.append(np.array(counts) - prev_count)
        # self.X = X
        # return None

        if self.args.random_noise:
            self.noise_amount = np.random.randint(low=self.args.offset, size=(len(X) - 1)) 
            if self.args.noise_test_index != -1:
                self.noise_amount = np.random.randint(low=5*self.args.noise_test_index, high=5*(self.args.noise_test_index+1), size=(len(X) - 1)) 
            if self.args.noise_high != -1:
                self.noise_amount = np.random.randint(low=self.args.noise_high-4, high=self.args.noise_high, size=(len(X) - 1)) 
        else:
            noise_amount = int(self.args.noise_ratio / 100 * self.args.offset)
            self.noise_amount = np.ones((len(X) - 1), dtype=int) * noise_amount

        if self.noise_amount.sum() == 0:
            X_noise = X[1:]
            event_counts = event_counts[1:]
        else:
            X_noise = []
            temp_count = []
            prev_events = X[0]
            prev_counts = event_counts[0]
            for i, noise_amount in enumerate(self.noise_amount):          
                if noise_amount == 0:
                    X_noise.append(X[i+1])
                    temp_count.append(event_counts[i+1])
                    prev_events = np.concatenate((prev_events[-self.args.offset:], X[i+1]))
                    
                    curr_count = event_counts[i+1] + prev_counts.max()
                    prev_counts = np.concatenate((prev_counts, curr_count))
                    continue
                
                noise = prev_events[-noise_amount:]
                prev_events = np.concatenate((prev_events[-self.args.offset:], X[i+1]))
                X_noise.append(np.concatenate((noise, X[i+1])))

                noise_count = prev_counts[-noise_amount:] - prev_counts[-noise_amount:].min() + 1
                curr_count = event_counts[i+1] + noise_count.max()
                temp_count.append(np.concatenate((noise_count, curr_count)))

                # print(prev_counts[:self.args.offset].max(), prev_counts[-self.args.offset:].min())
                prev_counts = prev_counts[-self.args.offset:] - prev_counts[-self.args.offset:].min() + 1
                count_ = event_counts[i+1] + prev_counts.max()
                prev_counts = np.concatenate((prev_counts, count_))
            event_counts = temp_count
            
        self.X = pad_sequences(X_noise, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len)  # B * T * V
        # print(Y[:10])
        self.prev_Y = np.array(Y[:-1])
        Y = np.array(Y[1:])
        self.org_Y = np.array(org_Y[1:])
        self.lengths = np.array(lengths[1:])
        idx_remove_short = np.where(self.lengths > 30)[0]
        self.lengths = self.lengths + self.noise_amount
        self.org_lengths = self.lengths.copy()
        self.lengths = np.where(self.lengths > self.args.seq_len, self.args.seq_len, self.lengths)
        
        # event_counts = event_counts[1:]
        
        event_counts = pad_sequences(event_counts, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len, value=0.0)
        count_max = np.reshape(np.max(event_counts, axis=1), (-1, 1))
        self.event_counts = np.where(event_counts != 0.0, event_counts, count_max)
        
        if self.args.remove_short:
            print("data shorter than 30 is excluded.")
            self.X = self.X[idx_remove_short]
            Y = Y[idx_remove_short]
            self.lengths = self.lengths[idx_remove_short]
            self.org_lengths = self.org_lengths[idx_remove_short]
            self.event_counts = self.event_counts[idx_remove_short]
            self.org_Y = self.org_Y[idx_remove_short]
            self.prev_Y = self.prev_Y[idx_remove_short]
            self.noise_amount = self.noise_amount[idx_remove_short]
            
        if self.args.with_other == False:
            print("Other class is excluded.")
            except_other = np.where(Y != 'Other')[0]
            self.X = self.X[except_other]
            Y = Y[except_other]
            self.lengths = self.lengths[except_other]
            self.org_lengths = self.org_lengths[except_other]
            self.event_counts = self.event_counts[except_other]
            self.org_Y = self.org_Y[except_other]
            self.prev_Y = self.prev_Y[except_other]
            self.noise_amount = self.noise_amount[except_other]
            
        self.idx2label = {i:label for i, label in enumerate(sorted(set(Y)))}
        self.label2idx = {label:i for i, label in self.idx2label.items()}
        self.Y = np.array([self.label2idx[l] for l in Y])
        if self.args.balance:
            self.balance_class()
        return None
        
    def event2matrix(self, sensors, values, timestamps, activities):
        activated = np.zeros(self.N_FEATURES)
        state_matrix, count_seq, labels = [], [], []
        prev_t = int(timestamps[0])
        prev_l = activities[0]
        count = 0
        for s, v, t, l in zip(sensors, values, timestamps, activities):
            t = int(t)            
            if t != prev_t:
                # label = l if prev_l == l else prev_l
                state_matrix.append(np.broadcast_to(activated, (t-prev_t, self.N_FEATURES)))
                count_seq += [count] * (t-prev_t)
                labels += [prev_l] * (t-prev_t)
                prev_t = t
            activated = self.change_state(activated, s, v)
            count += 1
            prev_l = l
        state_matrix.append(np.reshape(activated, [1, -1]))
        count_seq.append(count)
        labels.append(l)
        return np.concatenate(state_matrix), np.array(labels), count_seq   
    
    def balance_class(self):
        num_bed = len(np.where(self.Y == self.label2idx['Bed_to_toilet'])[0])
        num_eat = len(np.where(self.Y == self.label2idx['Eat'])[0])

        idx_bathing = np.where(self.Y == self.label2idx['Bathing'])[0]
        idx_cook = np.where(self.Y == self.label2idx['Cook'])[0]
        idx_bathing_chosen = np.random.choice(idx_bathing, size=num_bed, replace=False)
        idx_cook_chosen = np.random.choice(idx_cook, size=num_eat, replace=False)

        excluded_bathing = set(idx_bathing) - set(idx_bathing_chosen)
        excluded_cook = set(idx_cook) - set(idx_cook_chosen)
        excluded = excluded_bathing | excluded_cook

        idx = set(range(len(self.Y))) - excluded
        idx = np.array(list(idx))

        self.X = self.X[idx]
        self.Y = self.Y[idx]
        self.org_Y = self.org_Y[idx]
        self.lengths = self.lengths[idx]
        self.org_lengths = self.org_lengths[idx]
        self.event_counts = self.event_counts[idx]
        self.prev_Y = self.prev_Y[idx]
        self.noise_amount = self.noise_amount[idx]
        print(f'The number of the instances: {len(self.Y)}')


class Lapras(CASAS_RAW_SEGMENTED):
    def __init__(self, args):
        self.args = args
        self.idx_noise_amount = 0
        self.main()
        
    def main(self):
        if self.args.dataset == 'lapras_kisoo':
            self.data_path = "../AIoTService/segmentation/dataset/testbed/npy/lapras/csv"
        elif self.args.dataset == 'lapras_norm':
            self.data_path = "./dataset/Lapras_normalized"
        elif self.args.dataset == 'lapras_stand':
            self.data_path = "./dataset/Lapras_standardized"
        if self.args.random_noise:
            self.data_path += '_ms'
        print(f'data_path: {self.data_path}')
            
        self.episodes, sensors = self.preprocessing()
        if self.args.random_noise:
            self.noise_amount = np.random.randint(low=self.args.offset, size=len(self.episodes))
        else:
            noise_amount = int(self.args.noise_ratio / 100 * self.args.offset)
            self.noise_amount = np.ones(len(self.episodes), dtype=int) * noise_amount
        self.sensors = sorted(sensors)
        self.N_FEATURES = len(self.sensors)
        self.sensor2index = {sensor: i for i, sensor in enumerate(self.sensors)}
        
        self.args.seq_len = int(self.args.seq_len * self.args.window_size)
        self.X, self.Y, self.org_lengths, self.event_counts = self.generateDataset()
        self.args.seq_len = int(self.args.seq_len / self.args.window_size)
        self.X = np.reshape(self.X, (self.X.shape[0], self.args.seq_len, self.args.window_size, self.N_FEATURES))
        self.X = np.mean(self.X, axis=2)

        self.lengths = np.ceil(self.org_lengths / self.args.window_size).astype(np.int64)
        self.lengths = np.where(self.lengths > self.args.seq_len, self.args.seq_len, self.lengths)
        self.nseries, _, _ = self.X.shape
        self.N_CLASSES = len(np.unique(self.Y))
        self.args.offset = int(self.args.offset / self.args.window_size)
        
    def preprocessing(self):
        episodes = []
        for wd in glob.glob(f"{self.data_path}/*"):
            activity = wd.split("/")[-1]
            # print(activity)
            filelist = sorted(glob.glob(f"{wd}/*.csv"))
            for file in filelist:
                df = pd.read_csv(file, header=None)
                # df = df.loc[df[0].str.contains(r"Mtest")]
                # df = df.loc[df[0].str.contains(r"seat") | df[0].str.contains(r"Mtest")]
                df[2] = df[2].apply(lambda x: str(x)[:10])
                epi = df.to_numpy()
                episodes.append(np.concatenate([epi, np.broadcast_to(np.array([activity]), (len(epi), 1))], axis=1))
            # episodes.append(np.array([np.concatenate((pd.read_csv(file, header=None).to_numpy(), np.broadcast_to(np.array([activity]), (len(pd.read_csv(file, header=None).to_numpy()),1))), axis=1) for file in filelist]))            
        episodes = np.array(episodes)

        sensors = set()
        for ep in episodes:
            sensors |= set(ep[:, 0])
        return episodes, sensors  
    
    def event2matrix(self, episode):
        activated = np.zeros(self.N_FEATURES)
        start_time = int(float(episode[0][2]))
        episode[:,2] = list(map(lambda x: int(float(x) - start_time), episode[:,2]))
        duration = int(episode[-1][2])
        if duration < 1:
            return None
        # state_matrix = np.zeros((int(episode[-1][2]) + 1, self.N_FEATURES))
        state_matrix = np.zeros((self.args.seq_len + self.args.offset, self.N_FEATURES))
        # if self.args.dataset == 'lapras':
        #     state_matrix -= 1
        prev_t = 0
        count = 0
        count_seq = np.zeros((self.args.seq_len + self.args.offset))
        for s, v, t, l in episode:
            t = int(t)
            if t >= self.args.seq_len + self.args.offset:  # Only for the episode whose sequence length exceeds the predefined seq_len
                t = self.args.seq_len + self.args.offset - 1
                state_matrix[prev_t:t] = activated
                count_seq[prev_t:t] = count
                activated = self.change_state(activated, s, v)
                break
            if t != prev_t:
                state_matrix[prev_t:t] = activated
                count_seq[prev_t:t] = count
                prev_t = t
            activated = self.change_state(activated, s, v)
            count += 1
        state_matrix[t] = activated
        count_seq[t] = count       
        
        if self.noise_amount[self.idx_noise_amount] != 0:
            i = self.args.offset - self.noise_amount[self.idx_noise_amount]
            state_matrix = np.array(state_matrix[i:])
            duration = duration - i
            count_seq = count_seq[i:] - count_seq[:i].max()
            count_seq = np.where(count_seq > 0, count_seq, 0)
            # count_seq = np.where(count_seq[i:] != 0, count_seq[i:] - count_seq[:i].max(), 0)
            t = t - i
        self.idx_noise_amount += 1
        return [state_matrix, l, duration+1, count_seq, t, start_time]


    # def preprocessing(self):
    #     episodes = []
    #     for wd in glob.glob(f"{self.data_path}/*"):
    #         activity = wd.split("/")[-1]
    #         # print(activity)
    #         filelist = sorted(glob.glob(f"{wd}/*.npy"))
    #         episodes.append(np.array([np.concatenate((np.load(file), np.broadcast_to(np.array([activity]), (len(np.load(file)),1))), axis=1) for file in filelist]))            
    #     episodes = np.concatenate(episodes)
        
    #     sensors = set()
    #     for ep in episodes:
    #         sensors |= set(ep[:, 0])
    #     return episodes, sensors        
    
    

# data2 = Lapras(args)

# data_casas1 = CASAS_RAW_NATURAL(args)
# data_casas1.X

# np.sum(data1.X != data2.X)

# aa = episodes['Presentation'] + episodes['Discussion'] + episodes['GroupStudy'] + episodes['Chatting']
# lengths = []
# for a in aa:
#     lengths.append(int(a[-1][-1]) - int(a[0][-1]))
# np.mean(lengths)

# np.percentile(lengths, [0, 25, 50, 75, 90], interpolation='nearest')



    # def transition_boundary(self, offset=21):
    #     tau = 2
    #     tmax = 5
    #     gt_boundary = []
    #     for tr_point in self.tr_points:
    #         t = np.linspace(0, tmax, offset)
    #         y = np.exp(-t/tau)
    #         y_rev = np.sort(y)
    #         boundary = np.concatenate((y_rev, y[1:]))
    #         if offset - tr_point - 1 >= 0:
    #             boundary = boundary[offset-tr_point-1:]
    #         else:
    #             zeros = np.zeros(tr_point-offset+1)
    #             boundary = np.concatenate((zeros, boundary))
    #         gt_boundary.append(boundary)
    #     gt_boundary = pad_sequences(gt_boundary, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len*2)  # B * T * V
    #     return gt_boundary

# import utils
# args = utils.create_parser()

# args.dataset = "milan"
# args.with_other = False

# args.noise_ratio = 0
# args.random_noise = False
# data_false = CASAS_RAW_NATURAL(args)
# # args.noise_ratio = 100
# args.random_noise = True
# data_true = CASAS_RAW_NATURAL(args)

# np.unique(data.noise_amount, return_counts=True)
# np.min(data.lengths)
# data.X[2][21]
# data.noise_amount[2]

# prev_count = data.event_counts[0][-3:] - data.event_counts[0][-3:].min() + 1
# prev_count.max()

# data.event_counts[-5][:10]
# data.noise_amount[2]



# a = np.array([1,2,3,4,5])
# a[-2:]

# idx = 19
# data_true.noise_amount[idx]
# data_false.event_counts[idx][:10]
# data_true.event_counts[idx][data_true.noise_amount[idx]:10]

# data_true.noise_amount[18]
# np.where(data_true.noise_amount == 0)[0].min()
# data_false.event_counts[33][:10]
# data_true.event_counts[33][:10]


# np.max(data_true.event_counts[11])
# np.argmax(data_true.event_counts[11])
# data_true.lengths[11]
# data_true.event_counts[7][:10]
# data_true.noise_amount[7]

# np.max(data_false.event_counts[11])
# np.argmax(data_false.event_counts[11])
# data_false.lengths[11]
# data_false.event_counts[7][:30]


# data_true.event_counts[2][2:12]
# data_true.event_counts[2][:10]
# data_false.event_counts[2][:10]

# i = 0
# zero=[]
# for j, (t, f) in enumerate(zip(data_true.event_counts, data_false.event_counts)):
#     x = data_true.noise_amount[j]
#     if x == 0:
#         zero.append(j)
#         continue
#     t_ = t[x:] - t[:x].max()
#     f_ = f[:-x]
#     # t_ = t[x:x+10] - t[:x].max()
#     # f_ = f[:10]
#     a = np.where(t_==f_,0,1).sum()
#     if a != 0:
#         i += 1
#         print('f')
# np.where(data_true.event_counts[zero] != data_false.event_counts[zero], 1, 0).sum()


# i=0
# for j, (t, f) in enumerate(zip(data_true.X, data_false.X)):
#     x = data_true.noise_amount[j]
#     if x == 0:
#         continue
#     t_ = t[x:]
#     f_ = f[:-x]
#     a = np.where(t_ != f_, 1, 0).sum()
#     if a != 0:
#         i+=1
#         print(a, j)
# np.where(data_true.X[zero] != data_false.X[zero], 1, 0).sum()



# data_true.noise_amount[7]
# x = data_true.noise_amount[7]
# t = data_true.X[7]
# f = data_false.X[7]
# t_ = t[x:]
# f_ = f[:-x]
# a = np.where(t_ != f_, 1, 0).sum()

# len(t_)
# len(f_)

# t_[1]
# f_[1]

# t[3]
# t[4]
# data_true.event_counts[2169][:10]
# data_false.event_counts[2169][:10]

# len(data_true.noise_amount)

# data_true.X[0]

# len(data.X)
# data.X[0][:,].shape
# data.X[1].shape
# set(data.Y)
# data.X.shape
# data.Y.shape
# data.lengths.shape
# data.event_counts.shape
# data.idx2label



# a = np.array(['Bed_to_toilet', 'Other', 'Sleep'])
# np.where(a != 'Other')[0]

# a = np.array([1,2,3,4,5,6,7])
# a[np.where(a == data.label2idx['Other'])[0]]


# X = data.X
# np.concatenate((data.X[0], data.X[1])).shape



# offset = 20
# noise_ratio = 100
# noise_amount = int(noise_ratio / 100 * offset)

# X_noise = []
# prev_events = X[0][-noise_amount:]
# for i in range(len(X)-1):
#     noise = prev_events[-noise_amount:]
#     prev_events = np.concatenate((noise, X[i+1]))
#     X_noise.append(prev_events)

# self.X = pad_sequences(pair, padding='post', truncating='post', dtype='float32', maxlen=self.args.seq_len)  # B * T * V
# Y = Y[1:]
# lengths = lengths[1:]
# event_counts = event_counts[1:]   



# i = 0

# l = np.array(data.lengths)
# (np.where(l > 2000, 1, 0) * l).sum() / sum(np.where(l > 2000, 1, 0))
# np.where(l == 16, 1, 0).sum()

# len(data.X[153])
# a = data.tr_points[153]
# b = data.lengths[153]
# data.X[153][a+b:].sum()

# data.tr_points[353]
# data.lengths[153]
# data.gt_boundary[353][533:583]







# args.dataset = "cairo"
# data1 = CASAS_RAW_SEGMENTED(args)
# data1.X.shape
# data1.Y.shape

# df_y = pd.DataFrame({"Y":data1.Y, "d": data1.Y})
# df_y.groupby("Y").count()



# data_beg = CASAS_Milan_beginning(args)
# len(data_beg.X)
# len(np.where(data_beg.lengths == 3)[0])
# 4235 - 12 - 25

# 4190 4239
# 4189 4243

# data = CASAS_RAW_SEGMENTED(args)
# len(data.episodes)
# len(data.X)

# sensors, values, timestamps, activities = data.preprocessing()
# sensors_, values_, timestamps_, activities_ = data_beg.preprocessing()
# len(sensors)
# np.sum(np.array(sensors) != np.array(sensors_))



# np.sum(data.Y != data_beg.Y)


# len(data.episodes)

# len(data_beg.X) + 25 + 12

# data_beg.idx2label
# data.idx2label

# len(np.where(data.lengths == 6)[0])
# data_beg.Y[np.where(data_beg.lengths == 3)]

# data.X[0][0]
# data.lengths[2]
# len(data.X[2])
# data.X[2][6]


# data_beg.X[0][0]
# data_beg.lengths[2]
# data_beg.X[2][8]



