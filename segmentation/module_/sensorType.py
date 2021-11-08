import numpy as np
import pandas as pd
import glob
import collections
from itertools import permutations
from matplotlib import pyplot as plt

from .info.testbed import change_sensor_name
from .densityRatio import DensityRatio as dr
from .readText import time_correction

dataset_list = [
    'Chatting68.csv',
    'Chatting71.csv',
    'Chatting75.csv',
    'Chatting93.csv',
    'Discussion30.csv',
    'Discussion33.csv',
    'Discussion38.csv',
    'GroupStudy11.csv',
    'GroupStudy15.csv',
    'GroupStudy20.csv',
    'Presentation56.csv',
    'Presentation76.csv',
    'Presentation81.csv',
    'Presentation87.csv',
    'Chatting69.csv',
    'Chatting72.csv',
    'Chatting80.csv',
    'Discussion27.csv',
    'Discussion31.csv',
    'Discussion34.csv',
    'Discussion39.csv',
    'GroupStudy12.csv',
    'GroupStudy17.csv',
    'GroupStudy21.csv',
    'Presentation62.csv',
    'Presentation78.csv',
    'Presentation83.csv',
    'Chatting70.csv',
    'Chatting73.csv',
    'Chatting82.csv',
    'Discussion29.csv',
    'Discussion32.csv',
    'Discussion35.csv',
    'GroupStudy10.csv',
    'GroupStudy14.csv',
    'GroupStudy18.csv',
    'GroupStudy29.csv',
    'Presentation69.csv',
    'Presentation80.csv',
    'Presentation86.csv',
]

def check_unique_sensors():

    total_set = set()

    for file_name in dataset_list:

        full_dir = f"./dataset/testbed/RevisionData/{file_name}"
        csv_file = pd.read_csv(full_dir, names = ['sensor', 'value', 'timestamp'], header=None)

        total_set = total_set | set(csv_file.sensor)
    
    sensors = sorted(total_set)    

    return sensors

def check_unique_sensors_preprocessed():

    sensors = set()
    context_by_sensors = collections.defaultdict(set)

    for filedir in glob.glob("./csv/preprocessing/*.npy"):
        
        data = np.load(filedir)

        for row in data:
            s, v, t = row
            context_by_sensors[s].add(v)

        sensors |= set(data[:,0])
    
    return context_by_sensors

def generate_pair_with_others():

    # 1. load data
    # 2. remove unnecessary data
    # 3. generate pairs
    # 4. 
    sensors = set()
    datasets = collections.defaultdict(list)

    for filedir in glob.glob("./csv/preprocessing/*.npy"):

        data = np.load(filedir)
        label = filedir.split("/")[-1].split(".")[0][:-2]

        preprocessed_data = np.array(
            [item for item in data if item[0][0] in ["M", "L"]]
        )
        
        sensors |= set(preprocessed_data[:,0])

        datasets[label].append(preprocessed_data)

    pair_list = list(permutations(list(datasets.keys()), 2)) + [(k, k) for k in datasets.keys()]

    pair_activities = collections.defaultdict(list)

    for l, r in pair_list:
        for ldata in datasets[l]:
            for rdata in datasets[r]:

                pair = time_correction(
                    np.concatenate((ldata, rdata)), len(ldata)
                )

                pair_activities[f"{l}-{r}"].append((len(ldata), pair))


    return pair_activities, sorted(sensors)




def generate_preprocessed_data():
    
    for file_name in dataset_list:

        full_dir = f"./dataset/testbed/RevisionData/{file_name}"
        df = pd.read_csv(full_dir, names = ['sensor', 'value', 'timestamp'], header=None)

        preprocessed_data = []

        df_npy = df.to_numpy()
        for i in range(len(df_npy)):
            s, v, t = list(df_npy[i])
            v, t = str(v), str(float(t)/1000.)
            # row = []
            dict_value = change_sensor_name[s]
            if dict_value != "":
                if len(dict_value) == 1:
                    s_ = dict_value[0]
                    preprocessed_data.append([s_, v, t])
                else:
                    s_, v_ = dict_value[0], dict_value[1]
                    preprocessed_data.append([s_, v_, t])
                # preprocessed_data.append(row)

        name = file_name.split(".")[0]
        preprocessed_data = np.array(preprocessed_data)
        print(preprocessed_data.shape)

        np.save(f"./csv/preprocessing/{name}.npy", preprocessed_data)

        # # np.savetxt(f"./csv/preprocessing/{name}.csv", preprocessed_data, delimiter = ',')
        # pd.DataFrame(preprocessed_data).to_csv(f"./csv/preprocessing/{name}.csv")



def check_preprocessed_sensors():

    time_interval = 60
    n_, k_ = 50, 10

    # sensors = set()
    for full_file_name in glob.glob("./csv/preprocessing/*.npy"):
        file_name = full_file_name.split("/")[-1].split(".")[0]

        npy_data = np.load(full_file_name)

        sound_list = collections.defaultdict(list)

        interval_list = collections.defaultdict(list)

        min_mtime, max_mtime = np.inf, 0
        for row in npy_data:
            s, v, t = row
            index = int(float(t)/time_interval)
            if s[0]=='S':
                sound_list[s].append((float(t), float(v)))
                interval_list[index].append([s, float(v), float(t)])
            if s[0]=='M':
                min_mtime = min(min_mtime, float(t))
                max_mtime = max(max_mtime, float(t))
        
        # interval_avg = []
        # for k, v in interval_list.items():
        #     total, cnt = 0, 0
        #     for row in v:
        #         s, v, t = row
        #         if s[0]=="S":
        #             total += v
        #             cnt += 1
        #     interval_avg.append(total/cnt)

        # plt.plot(range(len(interval_avg)), interval_avg)

        
        for name, v in sound_list.items():
            x_, y_ = [a for a, _ in v], [b for _, b in v]

            scores = []

            for i in range(0, len(y_), n_):
                st, ed = i, i+2*n_+k_-1
                if ed >= len(y_):
                    break
                features = y_[st:ed]

                t0 = tn = []
                for j in range(n_):
                    t0.append(features[j:j+k_])
                    tn.append(features[n_+j:n_+j+k_])

                t0, tn = np.array(t0), np.array(tn)

                dre = dr(t0, tn, alpha=0.1)

                scores.append(dre.PEDiv)
            
            scores = np.array(scores)
            scores[scores < 0] = 0

            plt.plot(range(len(scores)), scores)
            plt.savefig(f"./analysis/testbed/sound/{name}.png")
            plt.clf()

            break
        
        break