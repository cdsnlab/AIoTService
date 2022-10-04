import dataclasses
from calendar import c
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader


# halting policy가 adaptive한지 확인하기 위함  ----------------------------------------------------------------
df = pd.read_csv('./output/log/220509-175942/fold_1/test_results.csv')
df['class'] = df['true_y'].map(lambda x: data.idx2label[x])

order = ['Bathing', 'Bed_to_toilet', 'Cook', 'Eat', 'Leave_Home', 'Relax', 'Sleep', 'Take_medicine', 'Work', 'Other']
fig, ax = plt.subplots()
plt.title(f"The number of the used events by classes ")
sns.set(rc = {'figure.figsize':(14,8)})
sns.barplot(y="class", x="event_count", data=df, ci="sd", ax=ax, order=order)
# plt.figure(figsize=(8,6))
plt.savefig('./analysis/used_events.png')
plt.clf()


fig, ax = plt.subplots()
plt.title(f"Waiting seconds by classes ")
sns.set(rc = {'figure.figsize':(14,8)})
sns.barplot(y="class", x="locations", data=df, ci="sd", ax=ax, order=order)
plt.xlabel("Waiting seconds")
# plt.figure(figsize=(8,6))
plt.savefig('./analysis/Waiting_seconds.png')
plt.clf()


# Kruskal-Wallis test. 
import scipy.stats as stats
stats.shapiro(df[df['true_y'] == 0]['event_count']) # 정규성 가정 만족 x

stats.kruskal(df[df['true_y'] == 0]['event_count'],
              df[df['true_y'] == 1]['event_count'],
              df[df['true_y'] == 2]['event_count'],
              df[df['true_y'] == 3]['event_count'],
              df[df['true_y'] == 4]['event_count'],
              df[df['true_y'] == 5]['event_count'],
              df[df['true_y'] == 6]['event_count'],
              df[df['true_y'] == 7]['event_count'],
              df[df['true_y'] == 8]['event_count'],
              df[df['true_y'] == 9]['event_count'])



# beginning of activity -----------------------------------------------------------------------------------

from heatmappy import Heatmapper

from PIL import Image


example_img_path = './analysis/heatmap/milan.png'
example_img = Image.open(example_img_path)

example_points = [(99, 174)] # 히트맵 중심 좌표 설정

# 히트맵 그리기
heatmapper = Heatmapper(
    point_diameter=90,  # the size of each point to be drawn
    point_strength=1,  # the strength, between 0 and 1, of each point to be drawn
    opacity=1,  # the opacity of the heatmap layer
    colours='default',  # 'default' or 'reveal'
                        # OR a matplotlib LinearSegmentedColorMap object 
                        # OR the path to a horizontal scale image
    grey_heatmapper='PIL'  # The object responsible for drawing the points
                           # Pillow used by default, 'PySide' option available if installed
)

# 이미지 위에 히트맵 그리기
example_img = heatmapper.heatmap_on_img(example_points, example_img)



example_points = [(208, 249)] # 히트맵 중심 좌표 설정

# 히트맵 그리기
heatmapper = Heatmapper(
    point_diameter=90,  # the size of each point to be drawn
    point_strength=0.5,  # the strength, between 0 and 1, of each point to be drawn
    opacity=1,  # the opacity of the heatmap layer
    colours='default',  # 'default' or 'reveal'
                        # OR a matplotlib LinearSegmentedColorMap object 
                        # OR the path to a horizontal scale image
    grey_heatmapper='PIL'  # The object responsible for drawing the points
                           # Pillow used by default, 'PySide' option available if installed
)

example_img = heatmapper.heatmap_on_img(example_points, example_img)
# 출력 이미지 경로 설정
example_img.save('./analysis/heatmap/heatmap_milan_test.png')


data.sensor2index.keys()



# example_img = Image.open(example_img_path)
# for i in range(example_img.size[0]): # x방향 탐색
#     for j in range(example_img.size[1]): # y방향 탐색
#         r, g, b = example_img.getpixel((i,j))  # i,j 위치에서의 RGB 취득
#         if r > g and r > b:
#             example_img.putpixel((i,j), (255, 255, 255))
#         else:
#             example_img.putpixel((i,j), (r, g, b))
# example_img.save('./analysis/heatmap_milan.png')


coord_sensors = {'D001': [(1038, 1102)], 
                 'D002': [(973, 978)], 
                 'D003': [(401, 1094)], 
                 'M001': [(1031, 1054)], 
                 'M002': [(1032, 894)], 
                 'M003': [(822, 953)], 
                 'M004': [(730, 307)], 
                 'M005': [(932, 38)], 
                 'M006': [(680, 103)], 
                 'M007': [(418, 156)], 
                 'M008': [(606, 306)], 
                 'M009': [(536, 532)], 
                 'M010': [(639, 811)], 
                 'M011': [(532, 813)], 
                 'M012': [(639, 983)], 
                 'M013': [(376, 503)], 
                 'M014': [(405, 1048)], 
                 'M015': [(406, 952)], 
                 'M016': [(428, 879)], 
                 'M017': [(470, 718)], 
                 'M018': [(351, 714)], 
                 'M019': [(474, 416)], 
                 'M020': [(144, 322)], 
                 'M021': [(87, 152)], 
                 'M022': [(563, 1023)], 
                 'M023': [(486, 959)], 
                 'M024': [(146, 901)], 
                 'M025': [(203, 532)], 
                 'M026': [(503, 155)], 
                 'M027': [(862, 647)], 
                 'M028': [(186, 229)]}



count_sensor = {label:{sensor: 0 for sensor in data.sensor2index.keys()} for idx, label in data.idx2label.items()}
for ep in data.episodes:
    s, _, _, l = ep[0]
    count_sensor[l][s] += 1
    
    # for s, _, _, l in ep:
    #     count_sensor[l][s] += 1


for cls, dic in count_sensor.items():
    img_path = './analysis/milan.png'
    img = Image.open(img_path).convert('RGB')
    max_count = max(dic.values())
    for sensor, coord in coord_sensors.items():
        if dic[sensor] == 0:
            continue
        heatmapper = Heatmapper(
            point_diameter=90,  # the size of each point to be drawn
            point_strength=dic[sensor] / max_count,  # the strength, between 0 and 1, of each point to be drawn
            opacity=0.65,  # the opacity of the heatmap layer
            colours='default',  # 'default' or 'reveal'
                                # OR a matplotlib LinearSegmentedColorMap object 
                                # OR the path to a horizontal scale image
            grey_heatmapper='PIL'  # The object responsible for drawing the points
                                # Pillow used by default, 'PySide' option available if installed
        )
        img = heatmapper.heatmap_on_img(coord, img).convert('RGB')
        r0, g0, b0 = img.getpixel((0,0))
    for i in range(img.size[0]): # x방향 탐색
        for j in range(img.size[1]): # y방향 탐색
            r, g, b = img.getpixel((i,j))  # i,j 위치에서의 RGB 취득
            if r == r0 and g == g0 and b == b0:
                img.putpixel((i,j), (255, 255, 255))
            else:
                img.putpixel((i,j), (r, g, b))
    # 출력 이미지 경로 설정
    img.save(f'./analysis/heatmap_{cls}_1.png')





# total = 0
# for k, v in count_sensor.items():
#     for s, count in v.items():
#         total += count


# Remove beginning of the activity ---------------------------------------------------------
df = pd.read_csv('./output/log/220513-174215/fold_1/test_results.csv')
df['event_count'].mean()


len_epi = []
for epi in data.episodes:
    if len(epi) <= 1:
        len_epi.append(len(epi))
    
len(len_epi)


args.prefix_len

len(data.episodes)
data.episodes[0][-args.prefix_len:, :]

suffix = []
for episode in data.episodes:
    converted = data.event2matrix(episode[-args.prefix_len:, :].copy())
    if converted is None:
        continue
    # event_counts.append(self.record_num_event(converted[0], converted[2]))
    suffix.append(converted[0][:converted[4]])



len(suffix)

suffix = np.array(suffix)
idx = np.random.choice(len(suffix), len(data.X))
suffix = suffix[idx]


data.X.shape
suffix.shape
np.concatenate((suffix, data.X), axis=1).shape


converted[0].shape
s = np.zeros([2,3])
d = np.ones([3,3])
sd = np.concatenate((s, d), axis=0)



from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences(sd, padding='post', truncating='post', dtype='float32', maxlen=3)  # B * T * V


np.random.choice(3, 1)


a = np.array([[0.1, 0.2, 0.7],
          [0.9, 0.0, 0.1]])

aa = tf.argmax(a, 1)
tf.reshape(aa, (-1, 1))

b = []
a = tf.convert_to_tensor(a)
b.append(a)
c = np.concatenate(b)


pad_sequences(c, padding='post', truncating='post', dtype='float32', maxlen=10, value=-1)



import pickle
with open('./output/log/220519-203251/fold_1/dict_analysis.pickle', 'rb') as f:
    data = pickle.load(f)
    
data.keys()
len(data['idx'])
len(data['raw_probs'])
len(data['all_yhat'])
len(data['true_y'])


a = []
for raw in data['raw_probs']:
    a.append(len(np.where(raw != -1)[0]))

a[0]
a[9]

data['raw_probs'][0][16]
data['raw_probs'][9][0]


# Load results from tensorboard.dev ----------------------------------------------
import tensorboard as tb

experiment_id = "LmnNS6b3SYainYUmSxPVbA"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df
# dfw = experiment.get_scalars(pivot=True) 
# dfw

df1 = df.loc[df['run'].str.contains(r"test")]
df1[df1["step"] == 99].groupby("tag").mean()



df = df[df["step"] == 100]

df["noise_proportion"] = df["run"].str.split('_').str[2]
df['noise_proportion'] = df['noise_proportion'].str.split('.').str[0]
df['noise_proportion'] = df['noise_proportion'].astype('int')
df = df[['value', 'noise_proportion']]
df = df.sort_values(by="noise_proportion")

import matplotlib.pyplot as plt
import pandas as pd
noise_proportion = df['noise_proportion']
Accuracy = df['value']
x=list(noise_proportion)
y=list(Accuracy)
plt.plot(x, y, color = 'b', linestyle = 'solid', marker = 'o', label = "Using complete data stream")
plt.xlabel('The proportion of the noise in the detected transition window (%)')
# plt.xticks(rotation = 25)
plt.ylabel('Accuracy')
plt.title('Accuracy by the proportion of the noise')
plt.legend()
plt.show()
plt.savefig('./analysis/noise_proportion.png')
plt.clf()


# To verify how much data is needed for adequate performances when the data is well-segmented
def acc_by_lam(experiment_ids, with_other):
    pd.options.display.float_format = '{:.2f}'.format
    # lam = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0]
    lam = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0]
    df_list = []
    for i, experiment_id in enumerate(experiment_ids):
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        df = experiment.get_scalars()
        df = df.loc[df['run'].str.contains(r"test")]
        df = df[df["step"] == 99]
        df = df[['tag', 'value']]
        df["lambda"] = lam[i]
        df_list.append(df)
    df_concat = pd.concat(df_list)
    df_concat = df_concat.pivot(index='lambda', columns='tag', values='value')
    df_concat = df_concat.sort_values(by="lambda", ascending = False)
    df_concat.index = df_concat.index.astype(str)
    df_concat = df_concat[['whole_accuracy', 'whole_count_mean', 'whole_earliness', 'whole_location_mean', 'whole_harmonic_mean']]
    df_concat.rename(columns={"whole_accuracy": "Accuracy", 
                            "whole_count_mean": "# used event", 
                            "whole_earliness": "Earliness", 
                            "whole_location_mean": "Waiting seconds",
                            "whole_harmonic_mean": "Harmonic mean"}, 
                    inplace=True)
    df_concat['Earliness'] = df_concat['Earliness'] * 100
    df_concat['Accuracy'] = df_concat['Accuracy'] * 100
    df_concat['Harmonic mean'] = df_concat['Harmonic mean'] * 100
    df_concat_sort = df_concat.sort_values(by="Earliness")

    x = df_concat_sort['Earliness']
    y = df_concat_sort['Accuracy']
    plt.plot(x, y, color = 'b', linestyle = 'solid', marker = 'o')
    
    # label = df_concat.index.to_list()
    # for xi,yi,li in zip(x,y,label):
    #     plt.annotate(li, # this is the value which we want to label (text)
    #                 (xi,yi), # x and y is the points location where we have to label
    #                 textcoords="offset points",
    #                 xytext=(0,10), # this for the distance between the points
    #                 # and the text label
    #                 ha='center',
    #                 arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.xlabel('Earliness (%)')
    # plt.xticks(rotation = 25)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Earliness')
    plt.legend()
    plt.show()
    plt.savefig(f'./analysis/Acc_byEarliness_{with_other}.png')
    plt.clf()

    return df_concat[['Accuracy', 'Earliness', 'Waiting seconds', '# used event', 'Harmonic mean']]

# experiment_ids_w_other = ['vvfpZu4xQAmpQ6wDHxTBOw','YivqHn6gQPi4uQ8ZWk6sDQ','QU1PmCx5Qem6pPpj1D04vA','1XszRYfaQRCXubrB2Pjaig','onNHRy80SDuYWsAYoKvjLA','pd4KvqPPTdOeu62kS3rqqg','fgmrdXs9TDOAlfQFJEiZyQ','2PRN4seSREqJ2Saf01jKtg','aPUKI1a8R7eWLfMasJ3hzA','71qmw4bQR0aFlBC8yIRHXw']
# experiment_ids_wo_other = ['A7oqlUEuRteRhbJyYRAZfg','gzZLlGroTAeSZWPt26V0mw','2eZoNUWZQOSyUwxcm31x1A','sMb8DC93RYmXMrZJPzTpiA','L30wvOFYSb2JvRtqf47aJA','0T4zC5BCReKeAM4Mvb9cKg','v0oGZndxRbOIlbKUJ2YUXw','2KOdOINrSveCn7WbYJ9yFQ','QAd5ilRXT7qVQAV2tpuR0Q','XFpbnJbhRp2UXOEkKtvTbA',]
experiment_ids_w_other = ['LmnNS6b3SYainYUmSxPVbA', 'UKtjkBHjQ8mXpuqEby7oQw', 'zhGjEhyCQmencpwb1dTzxw', 'meWeTlY9SVa0v8m6DV96rw', 'X4fL4sVHRteUobKNTaVNLg', '5ZfZa3TzQAqdlXVExqEX0w', 'iyeN3yLvT5iyNSyQYvZe1A', 'HKlNK3b4TLujFzn12NnCjg', 'Grazl2KDSJqplSMW2G8Asg', 'hOyGnw4SRyOFGn55Rq46eQ']
experiment_ids_wo_other = ['HrNAo5agSVqWF1x7PPdvKQ', 'FWurLNgbTXKWaJNYsK48bw', 'yMzgijl1SNG3J7smubojFg', 'dzokq0AjTnaxgW4H8D6tpg', 'X2jimn8VQyW1V4GqYwjR6w', 'r5Ia9ANHRwyzJSqCVsAHYg', 'KKc0lkELQS2Ou7vqRAnHmw', 'rkio7bLDR9mtxKoiqpKPVg', '4NJUYUXgRkqYjnWkW4GroQ', 'RdM7DR4bSVqiBuntrlM1Og', 'zxZt8w9sQRGKqmg9F6aT1A']

df_w_other = acc_by_lam(experiment_ids_w_other, 'w_other')
df_wo_other = acc_by_lam(experiment_ids_wo_other, 'wo_other')



# -----------------------------------------------------------------------------------------------------------------
# Data characteristics
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import *

data_name = "kyoto7"
args.dataset = data_name
args.with_other = False
data = CASAS_RAW_NATURAL(args)
np.max(data.org_lengths)

unique, counts = np.unique(data.lengths, return_counts=True)
for u, c in zip(unique, counts):
    print(f'activity:{u}, count:{c}')
    if u > 20:
        break

# The amount of each activity
data.org_Y.shape
data.Y.shape
unique, counts = np.unique(data.org_Y, return_counts=True)
for u, c in zip(unique, counts):
    print(f'activity:{u}, count:{c}')


# # Dataset hyperparameters
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--ntimesteps", type=int, default=2000, help="Dataset can control the number of timesteps")
# args = parser.parse_args()


# 데이터 로드 및 전처리 
# data.lengths = np.where(data.lengths > 2000, 2000, data.lengths)
# data.lengths.mean()

df = pd.DataFrame({"class":data.Y, "duration": data.org_lengths})
df['class'] = list(map(lambda x: data.idx2label[x], df['class']))
df.groupby("class").mean()

# df.groupby("class").mean().to_numpy()
df = df[df["duration"] > 0]
df["duration"].describe()
df["duration"].quantile(0.90)
df = df[df["duration"] < 2897.5]
df["duration"] = df["duration"] / 60
df["duration"].describe()

# 전체 데이터에 대한 히스토그램
plt.hist(df["duration"], bins=50)
plt.title("Histogram of activity duration")
plt.xlabel("Duration (min)")
plt.ylabel("Frequency")
plt.show()
plt.savefig(f"./analysis/data_length/hist_duration_{data_name}.png")
plt.clf()


# 클래스 별 박스플롯
df = df.sort_values(by='class' ,ascending=True)
df = pd.concat([df[df["class"] != "Other"], df[df["class"] == "Other"]])

sns.boxplot(y="class", x="duration", data=df, orient="h")
plt.title("Boxplot of duration by activity class")
plt.show()
plt.xlabel("Duration (min)")
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig(f"./analysis/data_length/boxplot_{data_name}.png")
plt.clf()


# 클래스 별 평균 길이 -----------------------------------------------------------------------
df.groupby("class").mean()


# raw data 및 state matrix 엑셀파일로 저장해서 확인  
# args.with_other = False
milan = CASAS_Milan(args)
type(milan.data)
type(milan.episodes)

milan.sensors[15]
pd.DataFrame(milan.data[1]).to_csv("./output/converted.csv")
pd.DataFrame(milan.episodes[1]).to_csv("./output/raw.csv")




# table 상에서 earliness와 time이 make sense하지 않은 이유 찾기 
df.groupby("class").mean()
(3.799 / df[df["class"] == "Other"]['duration']).mean()
other = df[df["class"] == "Other"]
other[other["duration"] < 40]

other["duration"].mean() * 0.1


data.lengths.shape
np.where(data.lengths <= 20)[0].shape
unique, counts = np.unique(data.Y[np.where(data.lengths < 10)[0]], return_counts=True)
unique, counts = np.unique(data.Y[np.where(data.lengths > 10)[0]], return_counts=True)


data.idx2label[8]



# confusion matrix
import os
import utils
args.with_other = True
args.balance = False
data = CASAS_RAW_NATURAL(args)
logdir = './output/log/backup/220723-013523'
with open(os.path.join(logdir, f'fold_{1}', 'dict_analysis.pickle'), 'rb') as f:
    dict_analysis = pickle.load(f)
dict_analysis.keys()
dict_analysis['all_yhat'].shape
dict_analysis['true_y'].shape
dict_analysis['all_yhat']


# all = np.array([[1,2,-1,-1,-1],
#               [1,3,5,-1,-1],
#               [1,3,5,1,1]])
a = np.where(dict_analysis['all_yhat'] == -1, 0, 1)
a = np.argmin(a, axis=1)
halt_pnt = np.where(a == 0, dict_analysis['all_yhat'].shape[1], a) - 1
halt_pnt = halt_pnt.reshape((-1, 1))
pred_y = np.take_along_axis(dict_analysis['all_yhat'], halt_pnt, axis=1).flatten()
pred_y = pred_y.astype(int)
np.where(dict_analysis['true_y'] == pred_y, 1, 0).mean()

dir = os.path.join(logdir, f'fold_{1}', 'confusion_matrix_real.png')
utils.plot_confusion_matrix(dict_analysis['true_y'], pred_y, dir, target_names=list(data.idx2label.values()))



# 액티비티 별 이동 경로
# on된 것 기준으로 경로 그리기
# 위 분석에서 차이가 없다면 duration이나 센서 간 상관관계에 대해 분석
# activity transition table 이전 activity가 feature가 될 수도 있지 않나?
import cv2
import numpy as np
from PIL import Image

from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader

args.dataset = "milan"
data = CASAS_RAW_SEGMENTED(args)

example_img_path = './analysis/heatmap/milan.png'
example_img = Image.open(example_img_path)

coord_sensors = {'D001': (1038, 1102), 
                 'D002': (973, 978), 
                 'D003': (401, 1094), 
                 'M001': (1031, 1054), 
                 'M002': (1032, 894), 
                 'M003': (822, 953), 
                 'M004': (730, 307), 
                 'M005': (932, 38), 
                 'M006': (680, 103), 
                 'M007': (418, 156), 
                 'M008': (606, 306), 
                 'M009': (536, 532), 
                 'M010': (639, 811), 
                 'M011': (532, 813), 
                 'M012': (639, 983), 
                 'M013': (376, 503), 
                 'M014': (405, 1048), 
                 'M015': (406, 952), 
                 'M016': (428, 879), 
                 'M017': (470, 718), 
                 'M018': (351, 714), 
                 'M019': (474, 416), 
                 'M020': (144, 322), 
                 'M021': (87, 152), 
                 'M022': (563, 1023), 
                 'M023': (486, 959), 
                 'M024': (146, 901), 
                 'M025': (203, 532), 
                 'M026': (503, 155), 
                 'M027': (862, 647), 
                 'M028': (186, 229)}


count_sensor = {label: [] for idx, label in data.idx2label.items()}
for ep in data.episodes:
    filtered = []
    for s, v, t, l in ep:
        if v == 'ON':
            filtered.append([s, v, t, l])
    count_sensor[l].append(np.array(filtered))


import itertools
import more_itertools
from collections import deque

def vis_moving_pattern(activity, seq_len, n_top):
    arrow = {}
    # activity = 'Bathing'
    # activity = 'Bed_to_toilet'
    # seq_len = 2
    # n_top = 20
    for i, epi in enumerate(count_sensor[activity]):
        queue_prev = deque()
        for j, (s, v, t, l) in enumerate(epi):
            if j < seq_len:
                queue_prev.append(s)
                continue
            sensor_chain = '_'.join(queue_prev)
            if sensor_chain in arrow:
                arrow[sensor_chain] += 1
            else:
                arrow[sensor_chain] = 1
            queue_prev.popleft()
            queue_prev.append(s)

    th = []
    im = cv2.imread('./analysis/heatmap/milan.png')
    max_count = max(list(arrow.values()))
    # for chain, count in arrow.items():
    top_seq = sorted(arrow.items(), key = lambda item: item[1], reverse=True)[:n_top]
    for chain, count in top_seq:
        sequence = chain.split('_')
        thickness = int(count/max_count*20+1)
        th.append(thickness)
        # if thickness == 0:
        #     continue
        for s1, s2 in more_itertools.pairwise(sequence):
            cv2.arrowedLine(im, coord_sensors[s1], coord_sensors[s2], (255,0,0), thickness)  
    cv2.imwrite(f'./analysis/moving_path/{activity}_{seq_len}_{n_top}.png', im)
    print(th)
    print(top_seq)

list_seq_len = [2, 3, 4, 5]
list_n_top = [20]
prod = [list_seq_len, list_n_top]
for seq_len, n_top in itertools.product(*prod):
    print(seq_len, n_top)
    vis_moving_pattern(activity='Bathing', seq_len=seq_len, n_top=n_top)
    vis_moving_pattern(activity='Bed_to_toilet', seq_len=seq_len, n_top=n_top)

vis_moving_pattern(activity='Bed_to_toilet', seq_len=5, n_top=20)


# -------------------------------------------------------------------------------------------
# earliness by lambda (bathing, bed_to_toilet)
import pandas as pd
import tensorboard as tb

experiment_ids = ['7lMvVMnoQnK2DAlQt8evOQ',
                '3lC7xQ0RQua8dg82S06HGA',
                'CxLNzLAoRX2AnQVMJSRVmg',
                'UeUjIPoNRqeQKJa3vp7Zsg',
                'yIMjI9zIRZ2qm214dzkdzw',
                'e6i8HR7nSjWWP120R7rynQ',
                'S9VNkKLAQFeDUDMWD138rg',
                'T2orDqnBSA2Y2v3mCRoeIg',
                'r4kIeMOyTeabuzuDJM1tZg',
                'xq4OjYV9Q5C0LAB1w3l9sg']

pd.options.display.float_format = '{:.2f}'.format
lam = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0]
df_list = []
for i, experiment_id in enumerate(experiment_ids):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df_bathing = df.loc[df['run'].str.contains(r"Bathing")]
    df_toilet = df.loc[df['run'].str.contains(r"Bed_to_toilet")]
    df = pd.concat([df_bathing, df_toilet])
    df = df[df["step"] == 49]
    df = df[['run', 'tag', 'value']]
    df["lambda"] = lam[i]
    df_list.append(df)
df_concat = pd.concat(df_list)

df_bathing = df_concat.loc[df['run'].str.contains(r"Bathing")]
df_toilet = df_concat.loc[df['run'].str.contains(r"Bed_to_toilet")]

df_bathing = df_bathing.pivot(index='lambda', columns='tag', values='value')
df_toilet = df_toilet.pivot(index='lambda', columns='tag', values='value')

df_bathing = df_bathing.sort_values(by="lambda", ascending = False)
df_toilet = df_toilet.sort_values(by="lambda", ascending = False)

df_bathing.index = df_bathing.index.astype(str)
df_toilet.index = df_toilet.index.astype(str)

df_bathing = df_bathing[['by_cls_event_count_mean']]
df_toilet = df_toilet[['by_cls_event_count_mean']]



# 다른 타입의 센서가 on 되기까지의 평균 term
from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader

args.dataset = "milan"
data = CASAS_RAW_SEGMENTED(args)
data.episodes

def calc_interval(data, target_class, subseq_len, same_type):
    # target_class = 'Bed_to_toilet'
    # target_class = 'Bathing'
    # subseq_len = None
    mean_interval = []
    for epi in data.episodes:
        if epi[0][-1] != target_class:
            continue
        prev_s = None
        prev_t = None
        interval = []
        for i, (s, v, t, l) in enumerate(epi):
            if i == subseq_len:
                break
            if v == 'OFF':
                continue
            if prev_s is None:
                prev_s = s
                prev_t = t
                continue
            if prev_s != s or same_type:
                interval.append(int(t)-int(prev_t))
                prev_s = s
                prev_t = t
        if interval != []:
            mean_interval.append(np.mean(interval))
    print(f'the number of episodes: {len(mean_interval)}')
    return np.mean(mean_interval), np.std(mean_interval)

subsequence_lengths = [2, 4, 6, 8, 10, 12, 14, None]
subsequence_lengths = [None]
for subseq_len in subsequence_lengths:
    # calc_interval(data=data, target_class='Bathing', subseq_len=subseq_len, same_type=True)
    calc_interval(data=data, target_class='Bathing', subseq_len=subseq_len, same_type=True)
    calc_interval(data=data, target_class='Bed_to_toilet', subseq_len=subseq_len, same_type=True)


# Duration of the activated sensors
from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader

args.dataset = "milan"
args.except_all_other_events=False

data = CASAS_RAW_SEGMENTED(args)

def calc_duration(data, target_class, subseq_len):
    mean_duration = []
    for epi in data.episodes:
        if epi[0][-1] != target_class:
            continue
        duration = []
        sensor_states = {s: -1 for s in data.sensor2index.keys()}
        for i, (s, v, t, l) in enumerate(epi):
            if i == subseq_len:
                break
            if v == "ON":
                sensor_states[s] = int(t)
            else:
                if sensor_states[s] != -1:
                    duration.append(int(t) - sensor_states[s])
                    sensor_states[s] = -1
        if duration != []:
            mean_duration.append(np.mean(duration))
    print(f'the number of episodes: {len(mean_duration)}')
    return np.mean(mean_duration), np.std(mean_duration)

subsequence_lengths = [2, 4, 6, 8, 10, 12, 14, None]
subsequence_lengths = [None]
for subseq_len in subsequence_lengths:
    calc_duration(data=data, target_class='Bathing', subseq_len=subseq_len)
    # calc_duration(data=data, target_class='Bed_to_toilet', subseq_len=subseq_len)


args.with_other=False
args.random_noise=True
args.except_all_other_events=True
data = CASAS_RAW_NATURAL(args)

args.except_all_other_events=False
data_other = CASAS_RAW_NATURAL(args)


count = np.zeros((data.N_FEATURES))
flag = np.zeros((data.N_FEATURES))
duration = []
for state in data.state_matrix:
    new_ON_idx = np.where((state==1) & (flag == 0))[0]
    new_OFF_idx = np.where((state==0) & (flag == 1))[0]
    # new_OFF_idx = np.where(((state==0) & (flag == 1)) | (count>=10))[0]
    
    flag[new_ON_idx] = 1
    flag[new_OFF_idx] = 0
    
    duration.append(count[new_OFF_idx])
    count[new_OFF_idx] = 0
    
    activated_idx = np.where(flag==1)[0]
    count[activated_idx] += 1
    
import time
for state in data_other.state_matrix:
    print(state)
    time.sleep(1)

    
duration = np.concatenate(duration)
duration.mean()
duration.std()
duration.max()
np.median(duration)



duration_0 = duration.copy()
duration_0.mean()
duration_0.std()
duration_0.max()

# 발생 시간대 -----------------------------------------------------------------------
import datetime
from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader

args.dataset = "milan"
data = CASAS_RAW_SEGMENTED(args)

def occurence_time(data, target_class):
    list_hours = []
    for t, y in zip(data.start_time, data.Y):
        label = data.idx2label[y]
        if label != target_class:
            continue
        list_hours.append(datetime.datetime.fromtimestamp(t).hour)
    return list_hours

hour_bathing = occurence_time(data, 'Bathing')
hour_toilet = occurence_time(data, 'Bed_to_toilet')

x = list(range(25))
y = []
for i in x:
    y.append(hour_toilet.count(i))



plt.bar(x, y)
plt.title('Bed_to_toilet')
plt.xlabel('Hour')
# plt.xticks(rotation = 25)
plt.ylabel('Frequency')
plt.legend()
plt.show()
plt.savefig(f'./analysis/ouccurence_hour_toilet.png')
plt.clf()



# -----------------------------------------------------------------------------------
# Analysis of the predicted class distribution 
import os

def entropy(p):
    id_p = np.where(p != 0)
    return -np.sum(p[id_p]*np.log(p[id_p]))

def entropy_by_activity(halt_pnt, activity='All'):
    if activity == 'All':
        idx = np.array(range(dict_analysis['true_y'].shape[0]))
    else:
        idx = np.where(dict_analysis['true_y'] == data_natural.label2idx[activity])[0]
    dist = dict_analysis['all_dist'][idx]
    halt_pnt = halt_pnt[idx].flatten()
    idx_wrong = np.where(pred_y[idx] != dict_analysis['true_y'][idx])[0]
    idx_correct = np.where(pred_y[idx] == dict_analysis['true_y'][idx])[0]
    dist_all = np.take_along_axis(dist, halt_pnt.reshape(-1, 1, 1), axis=1).squeeze().reshape(-1, len(data_natural.label2idx))
    dist_wrong = np.take_along_axis(dist[idx_wrong], halt_pnt[idx_wrong].reshape(-1, 1, 1), axis=1).squeeze().reshape(-1, len(data_natural.label2idx))
    dist_correct = np.take_along_axis(dist[idx_correct], halt_pnt[idx_correct].reshape(-1, 1, 1), axis=1).squeeze().reshape(-1, len(data_natural.label2idx))
    mean_entropy_wrong = np.mean([entropy(d) for d in dist_wrong])
    mean_entropy_correct = np.mean([entropy(d) for d in dist_correct])
    print(f'mean_entropy_wrong: {mean_entropy_wrong}\nmean_entropy_correct: {mean_entropy_correct}')
    return [entropy(d) for d in dist_all], [entropy(d) for d in dist_wrong], [entropy(d) for d in dist_correct]

args.dataset = "milan"
args.with_other = False
args.balance = True
data_natural = CASAS_RAW_NATURAL(args)

activities = list(data_natural.label2idx.keys()) + ['All']
logdir = './output/log/220820-115916'
# concat_entropy_wrong, concat_entropy_correct = [], []
concat_entropy_all = {a: [] for a in activities}
concat_entropy_wrong = {a: [] for a in activities}
concat_entropy_correct = {a: [] for a in activities}
concat_true_y, concat_pred_y, concat_halt_prob, concat_halt_pnt = [], [], [], []
for i in range(5):
    with open(os.path.join(logdir, f'fold_{i+1}', 'dict_analysis.pickle'), 'rb') as f:
        dict_analysis = pickle.load(f)
        print(dict_analysis.keys())
    a = np.where(dict_analysis['all_yhat'] == -1, 0, 1)
    a = np.argmin(a, axis=1)
    halt_pnt = np.where(a == 0, dict_analysis['all_yhat'].shape[1], a) - 1
    halt_pnt = halt_pnt.reshape((-1, 1))
    pred_y = np.take_along_axis(dict_analysis['all_yhat'], halt_pnt, axis=1).flatten().astype(int)
    halt_prob = np.take_along_axis(dict_analysis['raw_probs'], halt_pnt, axis=1).flatten()
    concat_true_y.append(dict_analysis['true_y'])
    concat_pred_y.append(pred_y)
    concat_halt_prob.append(halt_prob)
    concat_halt_pnt.append(halt_pnt)
    for activity in activities:    
        entropy_all, entropy_wrong, entropy_correct = entropy_by_activity(halt_pnt, activity)
        concat_entropy_all[activity] += entropy_all
        concat_entropy_wrong[activity] += entropy_wrong
        concat_entropy_correct[activity] += entropy_correct

raw = []
for (k_w, v_w), (k_c, v_c) in zip(concat_entropy_wrong.items(), concat_entropy_correct.items()) :
    print(f'{k_w}(wrong): {np.mean(v_w)}')
    print(f'{k_c}(correct): {np.mean(v_c)}')
    raw.append([k_w, np.mean(v_w), np.mean(v_c)])

df = pd.DataFrame(raw, columns=["Activity","entropy_wrong","entropy_correct"])
df.plot(x="Activity", y=["entropy_wrong", "entropy_correct"], kind="bar",figsize=(9,8), color=['orange','green'])
plt.xticks(rotation=45, ha="right")
plt.show()
plt.savefig('./analysis/bar_entropy.png')
plt.clf()

raw = []
for (k_w, v_w), (k_c, v_c) in zip(concat_entropy_wrong.items(), concat_entropy_correct.items()) :
    raw.append([k_w, np.mean(v_w + v_c)])

df = pd.DataFrame(raw, columns=["Activity","entropy"])
df.plot(x="Activity", y=["entropy"], kind="bar",figsize=(9,8), color=['blue'])
plt.xticks(rotation=45, ha="right")
plt.show()
plt.savefig('./analysis/bar_entropy_all.png')
plt.clf()


mean_acc = []
for true_y, pred_y in zip(concat_true_y, concat_pred_y):
    mean_acc.append(np.where(true_y == pred_y, 1, 0).mean())
np.mean(mean_acc)

concat_true_y = np.concatenate(concat_true_y)
concat_pred_y = np.concatenate(concat_pred_y)
concat_halt_pnt = np.concatenate(concat_halt_pnt)
np.where(concat_true_y == concat_pred_y, 1, 0).mean()


dir = os.path.join(logdir, f'fold_{1}', 'confusion_matrix_real.png')
utils.plot_confusion_matrix(concat_true_y, concat_pred_y, dir, target_names=list(data_natural.idx2label.values()))


concat_true_y = concat_true_y.reshape(-1, 1)
concat_pred_y = concat_pred_y.reshape(-1, 1)
cls_4 = np.array([0, 1, 2, 3])
target_class = np.where(concat_true_y == cls_4, 1, 0).sum(axis=1).reshape(-1, 1)
target_idx = np.where(target_class == 1)[0]

acc_target = np.where(concat_true_y[target_idx] == concat_pred_y[target_idx], 1, 0).mean()
location_target = concat_halt_pnt[target_idx].mean() + 1
print(f'Accuracy for target classes: {acc_target}')
print(f'Timesteps for target classes: {location_target}')




# ------------------------------------------------------------------------------------------
# Correlation between halting probability and entropy
import scipy.stats as stats

concat_true_y = np.concatenate(concat_true_y)
concat_pred_y = np.concatenate(concat_pred_y)
concat_halt_prob = np.concatenate(concat_halt_prob)

activity = "Eat"
if activity == "All":
    idx_selected = np.array(range(len(concat_true_y)))
else:
    idx_selected = np.where(concat_true_y == data_natural.label2idx[activity])[0]
selected_true_y = concat_true_y[idx_selected]
selected_pred_y = concat_pred_y[idx_selected]

idx_correct = np.where(selected_true_y == selected_pred_y)[0]
idx_wrong = np.where(selected_true_y != selected_pred_y)[0]

concat_halt_prob[idx_correct].mean()
concat_halt_prob[idx_wrong].mean()

stats.pearsonr(concat_entropy_all[activity], concat_halt_prob[idx_selected])
# stats.spearmanr(concat_entropy_all[activity], concat_halt_prob[idx_selected])



# ------------------------------------------------------------------------------------------
# 검증을 위한 정규성 및 등분산성 확인
import scipy.stats as stats

activity = 'All'
stats.shapiro(concat_entropy_wrong[activity] + concat_entropy_correct[activity]) # 정규성 가정 만족 x
stats.shapiro(concat_halt_prob[idx_selected]) # 정규성 가정 만족 x
stats.levene(concat_entropy_wrong[activity], concat_entropy_correct[activity]) # 등분산성 가정 만족 o

plt.figure(figsize=(8,6)) # 정규성 확인을 위한 시각화
stats.probplot(concat_entropy_wrong[activity] + concat_entropy_correct[activity], dist=stats.norm, plot=plt)
stats.probplot(concat_halt_prob[idx_selected], dist=stats.norm, plot=plt)
plt.savefig('./analysis/normal_test.png')
plt.clf()

stats.ranksums(concat_entropy_wrong[activity], concat_entropy_correct[activity]) # 2개 비교집단, non-parametric(분포에 대한 가정 x)
# stats.kruskal(concat_entropy_wrong[activity], concat_entropy_correct[activity]) # 3개 이상의 비교 집단, non-parametric(분포에 대한 가정 x)
np.mean(concat_entropy_wrong[activity])
np.mean(concat_entropy_correct[activity])




# def entropy(p):
#     id_p = np.where(p != 0)
#     return -np.sum(p[id_p]*np.log(p[id_p]))



# a = np.array([[0.1, 0.2, 0.3, 0.4],
#                 [0.3, 0.1, 0.4, 0.2],
#                 [0.4, 0.2, 0.1 ,0.3]])
# -np.sum(a*np.log(a), axis=1)



# -----------------------------------------------------------------------------------------
# 노이즈 테스트 
# dir_name: 220901-201847
import os
import pickle

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

import utils


def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def entropy(p):
    id_p = np.where(p != 0)
    return -np.sum(p[id_p]*np.log(p[id_p]))

def calc_results(logdir, correct_answer):
    # logdir = './output/log/220905-174712'
    with open(os.path.join(logdir, f'fold_{1}', 'dict_analysis.pickle'), 'rb') as f:
        dict_analysis = pickle.load(f)
    dict_analysis.keys() # dict_keys(['idx', 'raw_probs', 'all_yhat', 'true_y', 'all_dist'])

    # 20 이후에 halt 됐는지 확인
    a = np.where(dict_analysis['all_yhat'] == -1, 0, 1)
    a = np.argmin(a, axis=1)
    halt_pnt = np.where(a == 0, dict_analysis['all_yhat'].shape[1], a) - 1
    halt_pnt = halt_pnt.reshape((-1, 1))
    pred_y = np.take_along_axis(dict_analysis['all_yhat'], halt_pnt, axis=1).flatten().astype(int)

    idx_offset = np.where(halt_pnt.flatten() >= 20)[0]
    if correct_answer:
        idx_flitered = np.where(pred_y == dict_analysis['true_y'])[0]
    else:
        idx_flitered = np.where(pred_y != dict_analysis['true_y'])[0]
    idx = np.array(list(set(idx_flitered) & set(idx_offset)))

    halt_pnt = halt_pnt[idx]
    raw_probs = dict_analysis['raw_probs'][idx]
    all_yhat = dict_analysis['all_yhat'][idx]
    true_y = dict_analysis['true_y'][idx]
    all_dist = dict_analysis['all_dist'][idx]
    # data.idx2label # {0: 'Bathing', 1: 'Bed_to_toilet', 2: 'Cook', 3: 'Eat', 4: 'Leave_Home', 5: 'Relax', 6: 'Sleep', 7: 'Take_medicine', 8: 'Work'}

    all_gap, all_gap_abs = [], []
    all_highest_p, all_target_p = [], []
    all_entropy, all_entropy_gap, all_entropy_gap_abs = [], [], []
    all_cossim, all_cossim_gap, all_cossim_gap_abs = [], [], []
    for t_y, dist in zip(true_y, all_dist):
        gap, gap_abs = [], []
        highest_p, target_p = [], []
        etrp, etrp_gap, etrp_gap_abs = [], [], []
        cossim, cossim_gap, cossim_gap_abs = [], [], []
        for i, dist_t in enumerate(dist[:args.offset]):
            highest_p.append(np.max(dist_t))
            target_p.append(dist_t[t_y])
            etrp.append(entropy(dist_t))
            if i == 0:
                prev_target_p = dist_t[t_y]
                prev_dist= dist_t.copy()
                continue
            gap.append(dist_t[t_y] - prev_target_p)
            gap_abs.append(np.abs(dist_t[t_y] - prev_target_p))
            etrp_gap.append(entropy(dist_t) - entropy(prev_dist))
            etrp_gap_abs.append(np.abs(entropy(dist_t) - entropy(prev_dist)))
            cossim.append(cos_sim(dist_t, prev_dist))
            prev_target_p = dist_t[t_y]
            prev_dist= dist_t.copy()
        all_gap.append(np.array(gap))
        all_gap_abs.append(np.array(gap_abs))
        all_highest_p.append(np.array(highest_p))
        all_target_p.append(np.array(target_p))
        all_entropy.append(np.array(etrp))
        all_entropy_gap.append(np.array(etrp_gap))
        all_entropy_gap_abs.append(np.array(etrp_gap_abs))
        all_cossim.append(np.array(cossim))
    all_gap = np.array(all_gap)
    all_gap_abs = np.array(all_gap_abs)
    all_highest_p = np.array(all_highest_p)
    all_target_p = np.array(all_target_p)
    all_entropy = np.array(all_entropy)
    all_entropy_gap = np.array(all_entropy_gap)
    all_entropy_gap_abs = np.array(all_entropy_gap_abs)
    all_cossim = np.array(all_cossim)

    dict_results = {}
    dict_results['all_gap'] = np.mean(all_gap, axis=0)
    dict_results['all_gap_abs'] = np.mean(all_gap_abs, axis=0)
    dict_results['all_highest_p'] = np.mean(all_highest_p, axis=0)
    dict_results['all_target_p'] = np.mean(all_target_p, axis=0)
    dict_results['all_entropy'] = np.mean(all_entropy, axis=0)
    dict_results['all_entropy_gap'] = np.mean(all_entropy_gap, axis=0)
    dict_results['all_entropy_gap_abs'] = np.mean(all_entropy_gap_abs, axis=0)
    dict_results['all_cossim'] = np.mean(all_cossim, axis=0)
    dict_results['raw_probs'] = np.mean(raw_probs[:, :args.offset], axis=0)
    return dict_results

correct_answer = False
args = utils.create_parser()
args.with_other = False
args.noise_ratio = 50
data = CASAS_RAW_NATURAL(args)
dir = ['./output/log/220905-174613', './output/log/220905-174642', './output/log/220905-174712', './output/log/220905-174741', './output/log/220905-174810', './output/log/220905-174839', './output/log/220905-174908', './output/log/220905-174937', './output/log/220905-175007', './output/log/220905-191308']
noise_ratio = range(10, 100, 10)

result_name = 'all_gap'
answer_type = [True, False]
for answer in answer_type:
    for logdir, ratio in zip(dir, noise_ratio):
        print(logdir)
        dict_results = calc_results(logdir=logdir, correct_answer=answer)

        x = range(len(dict_results[result_name]))
        y = dict_results[result_name]
        plt.figure(figsize=(8,6))
        plt.plot(x, y, color = 'b', linestyle = 'solid', marker = 'o')
        plt.xlabel('Timestep')
        plt.ylabel('Probability gap for true class')
        plt.title(f'Noise {ratio}%')
        plt.xticks(range(0, 20, 2), range(0, 20, 2)) 
        plt.legend()
        plt.show()
        plt.savefig(f'./analysis/noise/{result_name}_{answer}_noise{ratio}.png')
        plt.clf()
        
for answer in answer_type:
    for logdir, ratio in zip(dir, noise_ratio):
        dict_results = calc_results(logdir=logdir, correct_answer=answer)
        df = pd.DataFrame({'timesteps':range(len(dict_results['all_highest_p'])),
                        'highest_prob': dict_results['all_highest_p'], 
                        'target_prob': dict_results['all_target_p']})

        df.plot(x="timesteps", y=["highest_prob", "target_prob"], kind="bar",figsize=(9,8), color=['orange','green'])
        # plt.xticks(rotation=45, ha="right")
        plt.title(f'Noise {ratio}%')
        plt.ylabel('Probability')
        plt.show()
        plt.savefig(f'./analysis/noise/all_prob_{answer}_noise{ratio}.png')
        plt.clf()
        
        
        
        
# noise ---------------------------------------------------------------------------------------------------
dir = './output/log/220908-170442/fold_3'
df = pd.read_csv(f'{dir}/test_results.csv')
df['locations'].mean()
np.where(df['true_y'] == df['pred_y'], 1, 0).mean()





# attention ------------------------------------------------------------------------
import pandas as pd
import tensorboard as tb


def get_exp_results(exp_ids, lam):
    df_list = []
    for i, experiment_id in enumerate(exp_ids):
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        df = experiment.get_scalars()
        df = df.loc[df['run'].str.contains(r"test")]
        df["step"] += 1
        df["group"] = df["tag"] + "_" + df["step"].astype('str')
        df_grouped = df.groupby("group").mean()
        
        df_grouped['metrics'] = list(map(lambda x: '_'.join(x.split('_')[:-1]) , df_grouped.index.to_list()))
        df_grouped = df_grouped.sort_values(by=['metrics', 'step'])
        df_grouped["lambda"] = lam[i]
        df_list.append(df_grouped)
    df_concat = pd.concat(df_list)
    return df_concat

# pd.options.display.float_format = '{:.2f}'.format
lam = [0.1, 0.01, 0.001, 0.0001, 0.00001]
exp_id_basic = ['v3jev14qS3CXdSlHXegPJg',
                '8uW7Mi7FQNm9qK9FAGtLuQ',
                'y5kYFpZMQmenJNmg9NumHA',
                'nLZJSkvxSXeAwIKZbv79KQ',
                'RIwwVbNyTVyWN59bslZgbQ']

# exp_id_attn = ['QX0YpanESrmQ8k2Zjr8ZaQ',
#                 '4btlIYCgSg2RWLbcNcZd6Q',
#                 'jg2QEk3qSlSGruN3m9Jveg',
#                 'mjoQytxkR6K88i19BoAVGg',
#                 'FL00TO5GREW7ayjNlVQBpw']

# exp_id_attn_l = ['0JfZOC7QRZ2rRzxePD6zFQ',
#                 'FJLQRZtOTweAl9cPoprU6g',
#                 'OmTV7s5mQuepF01GCM192g',
#                 'YSa0q6E8SRK0x2Pravre5w',
#                 'H4uDieYDQhKdFpjdQxVT7A']

# exp_id_attn_drop = ['EahXBaPDQsaXmJt7b4kcdA',
#                     'X7hXJGzRSBqC6Se12IogFA',
#                     'bUdOu0gWQ0Wv1CXJgVR5Nw',
#                     'FNoLX85LSEGYzJQKVcXEOw',
#                     'ho5WWDTES8SLvrpF7NoINQ']

exp_id_attn_drop = ['ZaVVDpQxREOOw249hPq20w',
                    '7ck9gCDmTBazMoPH6S6srw',
                    'TeB0D8vrTY6D8E1o9FdQwQ',
                    'QgDqRCXKSau71jWg8ceO1g',
                    'K09RhE46SLmDLYogqTkeag']

# exp_id_attn_tr_points = ['S0ePzk5PRWGJHVqlZ7S6Ow',
#                         'hpFezAAvTwibRRq29TeWuQ',
#                         'QlYJpeByQYOZoLB7Nwmwqw',
#                         'RlSeOE1ZRCmtVXWxVdA1tw',
#                         'Ko2cqmLTR4KWFe2Zl4pRqA']

exp_id_attn_tr_points = ['16s8Z5NESBOHa0lLOjyH5w',
                        '4pOeDQKsQ8CO19xAAIe5sQ',
                        '4ILAlSvWSq6XsYt8mGZJzg',
                        'PnkbjclaRPOJZl64sxqumg',
                        'dA57yRfdTPe5sQpcKRWSOg']


# exp_id_basic_other = ['1zbqK6zDQM6DGdRf1C4x4Q',
#                     'I70nJjM0RZOOq3xCLK7f5w',
#                     'fZeBtQrGR0WNJsKVnRWlhQ',
#                     'LbtW8N08Q7OMopAZgbWNhQ',
#                     'cvxbUkiySRerLrpzXmAy3A']

# exp_id_attn_other = ['LqN8D1AnSy6RqpHGlXMwMQ',
#                     'GCzHZOE1R5mRVpZfd2e3bg',
#                     'o96GQH2bQMqNVh9fASy02A',
#                     'HLNMmCdpSmq1wDANjg9mUQ',
#                     'pLnpcp7BRs2jENVpldTBjw']

# exp_id_attn_tuning = ['ERpXUnkxQNuEhBZy19vjSA',
#                     '7gRN31DHQouEHqd7cmJGtQ',
#                     'bipqP93cQ4y6iMvohRCOYQ',
#                     'ZpI16eEnQBWubZKBDmCWSg',
#                     'glS63TJ7TxCg7d8I5ozpxQ',
#                     'e3qwHJedSJSkGVGuLiLJsA',
#                     'CWeX4NvPSKam9PYivpWJrA',
#                     'RbVUlINxTzur776KFo1POA',
#                     'XJv8qd3ZQTmpsrJ6MQ6DkQ']

# exp_id_basic_halting = ['WXA992uxSkuq14AQQSnZmA',
#                         'eePdxf15QWSLEg8ULZKMKg',
#                         'I5tBSbMpRkCqQaclgw0bSw',
#                         'W0AzDsxFRjOlPevdGY0I3g']

exp_id_noise_ratio = ['WAOy8JjNSAunfvQmxN0LSQ',
                        'W9d5Rd13TLmJDU1L4N7OUw',
                        'm8QHATYURySLdwyuANIExA',
                        'K47AelJYS0Onn9B3jS2NWQ',
                        '7lVZe75pQK2ZKwjJZLATWQ',
                        'X2WkYUxMRMGlnqDDhjqU4A',
                        'R1gesXloS7GLpeAsZuOeTw',
                        'zn0n4JrlTvqY6CEVzETJNg',
                        'jgKE4vhRQG2lgKvecoQbfg',
                        'vg1Ar3ztRVavl53Og8LWUw',
                        'aQFMiGZVRYWENWAxeWIjXQ',
                        'mxTB9tvjRoK7DKnhruzDmQ']

exp_id_noise_ratio_cnn = ['CEr0svYASmmxm6L01aw9MA',
                        'I0GzkFEwQTO7uDlzJoIwzQ',
                        'zEGjEdcZQ5qeFGtebCnejA',
                        'ZElc9y2rQ2qtZvVmcGBhDQ',
                        'r0P2LRJYTXm0QA8JaQZCqQ',
                        'OKvBfe31RpywRbhpQAbpNA',
                        'xYwo30n7TNKdqV9asWNedw',
                        '0F9typOVTKeVvGUnwQnRIA']


exp_id_excl_other = ['PIaOMGxLQ3ujfrkV8SLWsQ',
                    'gLL7epEpT2G093jNJMFW8w',
                    'vnY0DyNYSai2ie5zdOlpmA',
                    'XLjAHAp1QdijBAqfkorjVA',
                    'uoquXU9ZQ46dJCJWjK2exw',
                    'o8ag24WxRIyxuzVIX3AwPw',
                    'GmW1YwknRqGrM793Wyed5g',
                    'DbbjgMgqTRGfOJd9vpsWwg',
                    '2YglPKjdQya1RpIzbbKoCQ',
                    'QYdUOLFxSV29BgHqpezPVQ']

# exp_id_attn_cls_token_1 = ['NMfAF46GQEqdFBd8Wztauw',
#                         'Fgf0UmzoTOCw2WaUBMVtDA',
#                         'QqBqte2FRzePjlfJGZqdFQ',
#                         'RyZn8K82QFetXUXTACmMSA',
#                         'TedklrWtRQiJ1tmUJQmibA']

exp_id_attn_cls_token_2 = ['bcvhlgmcQzaQqxIYRwYDTQ',
                            'cwky9l3WTvuru99NPHdTvg',
                            'u0FGxkZuQJGxxJ4EjGywSw',
                            'yBdbD6XdQIKrGKSBe4DxQw',
                            'jZoU1SdBTQOq78da4BHkVQ']

exp_id_cnn = ['Buxl1DMqTKG4a45Y6vSWFA',
            'C6fyA6QDSQ6YldJh7s0ueA',
            'X1yQXniXRW2xr0H9MGe3JA',
            'Qu7Y0cXVSDWEG06xXf0Etg',
            '5ckJyqpLR66gZFR69Ea2uw']

exp_id_noise_relation = ['m4eQ3FGmSgSKikq4c9E4Dw',
                        'xuN6Hl4SQo20KpAr8AIrCA',
                        'XoXLEypOSJ6cdlyqktNqAg',
                        'ETpMF0b8Qv67OLdUJPmK3w',
                        '4LgOOl5YQeKaIKeUxWKT0g',
                        'R73ZoXUDTeKQmmvnX9SVXQ',
                        'Y40LI69sQFKy9NAJmcDWzQ',
                        'fLGTeDe1Ra2WYFGSbLI4Ag',
                        'pKu1VPyhQPazRrjnUllk9Q',
                        'G1nqfxGaRwK2McR4Zfnocw',
                        'T1JFX0MWQvS2zXpVqU6p8Q',
                        'eujZj3GORG2cjYLC4E3flw',
                        'ouQ2eZvnQc2zMhyA9rabvQ',
                        'cNflIvlaRMu8kokQkolmDQ',
                        '6Y0VG69ORpKXTgvtk05YOQ',
                        'Lc8szfK3TRevgi3Gj4fOaw',
                        'dbR5cb2UQsym2l7iJMGPFw',
                        'Sn1cLc0vQceoAb0KoKoWOQ',
                        'VoVZyqWKQLG7Gg36ZurcwQ',
                        '5750Cj2oQm6UwmYOFlWw7Q',
                        'xVWtpw1vQpe6R8PwqIPjjg',
                        'OZaYRrSGTpOgwJaU9jlQGA',
                        'UNYOSVF4QXmqobZJHnhwrg',
                        'xWWAAIRhQL6KNPYaiqw8ug',
                        '9WcZLN3uQpiKYEoaq5R8Ng',
                        'LHzqXTAvS5GOzTbI3jkU0Q',
                        'gPLZoyMxRImP4Najoey8Vg',
                        'peu1W3wGT9GErRzrOYQuJg',
                        'G8ZBBTmnRHOEYhzq5lOgYA',
                        'hPUmRy0XQj64pFPgppKjng',
                        'KGFp7r5BSwuNmOy1gH0ZPg',
                        '6ujFACXjRSKRSbP6X0tvTw',
                        'bxT9DZ0WQQ6vW4PPAnO84g',
                        'xHxksZTBT0aC1MM3Pj5OXQ',
                        'dfN6lWjqSzmGns5zer7exQ',
                        'niW9LfGQQeqG7M0GOxS5Aw',
                        'M0KXD6jTRcmLbgPE7TgUnw',
                        'qbh6nAeNQ8WtV2PJxwCAgQ',
                        'Vaqv3L1OTOiou9IvfaDG4w',
                        'BV4mO8nfQyKCvd4CrPQbRg']


# lam = [0.1, 0.2, 0.3, 0.4, 0.5]
lam=['0.1', '0.01', '0.001', '0.0001', '0.00001']
df_attn = get_exp_results(exp_id_attn, lam)
df_attn_l = get_exp_results(exp_id_attn_l, lam)
df_basic = get_exp_results(exp_id_basic, lam)
df_attn_drop = get_exp_results(exp_id_attn_drop, lam)

df_basic_other = get_exp_results(exp_id_basic_other, lam)
df_attn_other = get_exp_results(exp_id_attn_other, lam)
df_attn_tr_points = get_exp_results(exp_id_attn_tr_points, lam)
# df_attn_cls_token_1 = get_exp_results(exp_id_attn_cls_token_1, lam)
df_attn_cls_token_2 = get_exp_results(exp_id_attn_cls_token_2, lam)
df_cnn = get_exp_results(exp_id_cnn, lam)

exp_num = ['31_1', '31_2', '31_3',
            '64_1', '64_2', '64_3',
            '128_1', '128_2', '128_3',]
df_attn_tuning = get_exp_results(exp_id_attn_tuning, exp_num)

df_basic_halting = get_exp_results(exp_id_basic_halting, lam)

exp_num = ['none_0', 'none_1', 'none_2', 'none_3',
            'basic_0', 'basic_1', 'basic_2','basic_3', 
            'Attn_0', 'Attn_1', 'Attn_2', 'Attn_3']
df_noise_ratio = get_exp_results(exp_id_noise_ratio, exp_num)

exp_num = ['cnn_0.1_0', 'cnn_0.1_1', 'cnn_0.1_2', 'cnn_0.1_3',
           'cnn_0.01_0', 'cnn_0.01_1', 'cnn_0.01_2', 'cnn_0.01_3']
df_noise_ratio_cnn = get_exp_results(exp_id_noise_ratio_cnn, exp_num)

exp_num = list(range(10))
df_excl_other = get_exp_results(exp_id_excl_other, exp_num)

from itertools import product
noise_high=['4', '6', '8', '10', '12', '14', '16', '18']
lam=['0.1', '0.01', '0.001', '0.0001', '0.00001']
exp_num = []
for i in product(noise_high, lam):
    exp_num.append('_'.join(i))
df_noise_relation = get_exp_results(exp_id_noise_relation, exp_num)



# df_attn[(df_attn['lambda'] == 0.0001) & (df_attn['metrics'] == 'whole_accuracy')]
# df_attn[(df_attn['lambda'] == 0.0001) & (df_attn['metrics'] == 'whole_earliness')]
# df_attn[(df_attn['lambda'] == 0.0001) & (df_attn['metrics'] == 'whole_harmonic_mean')]

df_attn_l[(df_attn_l['lambda'] == '0.01') & (df_attn_l['metrics'] == 'whole_accuracy')]
df_attn_l[(df_attn_l['lambda'] == '0.01') & (df_attn_l['metrics'] == 'whole_earliness')]
df_attn_l[(df_attn_l['lambda'] == '0.001') & (df_attn_l['metrics'] == 'whole_harmonic_mean')]

df_basic[(df_basic['lambda'] == 0.01) & (df_basic['metrics'] == 'whole_accuracy')]
df_basic[(df_basic['lambda'] == 0.1) & (df_basic['metrics'] == 'whole_earliness')]
df_basic[(df_basic['lambda'] == 0.00001) & (df_basic['metrics'] == 'whole_harmonic_mean')]

df_attn_drop[(df_attn_drop['lambda'] == 0.01) & (df_attn_drop['metrics'] == 'whole_accuracy')]
df_attn_drop[(df_attn_drop['lambda'] == 0.01) & (df_attn_drop['metrics'] == 'whole_earliness')]
df_attn_drop[(df_attn_drop['lambda'] == 0.01) & (df_attn_drop['metrics'] == 'whole_harmonic_mean')]

# df_basic_other[(df_basic_other['lambda'] == 0.00001) & (df_basic_other['metrics'] == 'whole_accuracy')]
# df_basic_other[(df_basic_other['lambda'] == 0.0001) & (df_basic_other['metrics'] == 'whole_earliness')]
# df_basic_other[(df_basic_other['lambda'] == 0.0001) & (df_basic_other['metrics'] == 'whole_harmonic_mean')]

# df_attn_other[(df_attn_other['lambda'] == 0.00001) & (df_attn_other['metrics'] == 'whole_accuracy')]
# df_attn_other[(df_attn_other['lambda'] == 0.00001) & (df_attn_other['metrics'] == 'whole_earliness')]
# df_attn_other[(df_attn_other['lambda'] == 0.00001) & (df_attn_other['metrics'] == 'whole_harmonic_mean')]

# df_attn_tuning[(df_attn_tuning['lambda'] == '64_1') & (df_attn_tuning['metrics'] == 'whole_accuracy')]
# df_attn_tuning[(df_attn_tuning['lambda'] == '64_1') & (df_attn_tuning['metrics'] == 'whole_earliness')]
# df_attn_tuning[(df_attn_tuning['lambda'] == '64_1') & (df_attn_tuning['metrics'] == 'whole_harmonic_mean')]

df_basic_halting[(df_basic_halting['lambda'] == 0.1) & (df_basic_halting['metrics'] == 'whole_accuracy')]
df_basic_halting[(df_basic_halting['lambda'] == 0.0001) & (df_basic_halting['metrics'] == 'whole_earliness')]
df_basic_halting[(df_basic_halting['lambda'] == 0.01) & (df_basic_halting['metrics'] == 'whole_harmonic_mean')]

id = 'none_3'
df_noise_ratio[(df_noise_ratio['lambda'] == id) & (df_noise_ratio['metrics'] == 'whole_accuracy')]
df_noise_ratio[(df_noise_ratio['lambda'] == id) & (df_noise_ratio['metrics'] == 'whole_earliness')]
df_noise_ratio[(df_noise_ratio['lambda'] == id) & (df_noise_ratio['metrics'] == 'whole_location_mean')]
df_noise_ratio[(df_noise_ratio['lambda'] == id) & (df_noise_ratio['metrics'] == 'whole_harmonic_mean')]

id = 'cnn_0.01_3'
df_noise_ratio_cnn[(df_noise_ratio_cnn['lambda'] == id) & (df_noise_ratio_cnn['metrics'] == 'whole_accuracy')]
df_noise_ratio_cnn[(df_noise_ratio_cnn['lambda'] == id) & (df_noise_ratio_cnn['metrics'] == 'whole_earliness')]
df_noise_ratio_cnn[(df_noise_ratio_cnn['lambda'] == id) & (df_noise_ratio_cnn['metrics'] == 'whole_location_mean')]
df_noise_ratio_cnn[(df_noise_ratio_cnn['lambda'] == id) & (df_noise_ratio_cnn['metrics'] == 'whole_harmonic_mean')]

id = 7
df_excl_other[(df_excl_other['lambda'] == id) & (df_excl_other['metrics'] == 'whole_accuracy')]
df_excl_other[(df_excl_other['lambda'] == id) & (df_excl_other['metrics'] == 'whole_earliness')]
df_excl_other[(df_excl_other['lambda'] == id) & (df_excl_other['metrics'] == 'whole_harmonic_mean')]


id = 0.1
df_attn_tr_points[(df_attn_tr_points['lambda'] == id) & (df_attn_tr_points['metrics'] == 'whole_accuracy')]
df_attn_tr_points[(df_attn_tr_points['lambda'] == id) & (df_attn_tr_points['metrics'] == 'whole_earliness')]
df_attn_tr_points[(df_attn_tr_points['lambda'] == id) & (df_attn_tr_points['metrics'] == 'whole_harmonic_mean')]

id = 0.01
df_attn_cls_token_1[(df_attn_cls_token_1['lambda'] == id) & (df_attn_cls_token_1['metrics'] == 'whole_accuracy')]
df_attn_cls_token_1[(df_attn_cls_token_1['lambda'] == id) & (df_attn_cls_token_1['metrics'] == 'whole_earliness')]
df_attn_cls_token_1[(df_attn_cls_token_1['lambda'] == id) & (df_attn_cls_token_1['metrics'] == 'whole_harmonic_mean')]

id = 0.01
df_attn_cls_token_2[(df_attn_cls_token_2['lambda'] == id) & (df_attn_cls_token_2['metrics'] == 'whole_accuracy')]
df_attn_cls_token_2[(df_attn_cls_token_2['lambda'] == id) & (df_attn_cls_token_2['metrics'] == 'whole_earliness')]
df_attn_cls_token_2[(df_attn_cls_token_2['lambda'] == id) & (df_attn_cls_token_2['metrics'] == 'whole_harmonic_mean')]

id = 0.01
df_cnn[(df_cnn['lambda'] == id) & (df_cnn['metrics'] == 'whole_accuracy')]
df_cnn[(df_cnn['lambda'] == id) & (df_cnn['metrics'] == 'whole_earliness')]
df_cnn[(df_cnn['lambda'] == id) & (df_cnn['metrics'] == 'whole_harmonic_mean')]

pd.options.display.float_format = '{:.4f}'.format
id = '4_0.00001'
df_noise_relation[(df_noise_relation['lambda'] == id) & (df_noise_relation['metrics'] == 'whole_accuracy')]
df_noise_relation[(df_noise_relation['lambda'] == id) & (df_noise_relation['metrics'] == 'whole_earliness')]
df_noise_relation[(df_noise_relation['lambda'] == id) & (df_noise_relation['metrics'] == 'whole_harmonic_mean')]



df_list = []
for i, experiment_id in enumerate(exp_ids):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df = df.loc[df['run'].str.contains(r"train")]
    
    df["group"] = df["tag"] + "_" + df["step"].astype('str')
    df_grouped = df.groupby("group").mean()
    
    df_grouped['metrics'] = list(map(lambda x: '_'.join(x.split('_')[:-1]) , df_grouped.index.to_list()))
    df_grouped = df_grouped.sort_values(by=['metrics', 'step'])
    df_grouped["lambda"] = lam[i]
    df_list.append(df_grouped)
df_concat = pd.concat(df_list)

df_concat

df_concat[(df_concat['lambda'] == 0.1) & (df_concat['metrics'] == 'whole_accuracy')]


# attention score 확인 ---------------------------------------------------------------------------


import pickle
import numpy as np

dir = '220926-154933'
all_noise_amount, all_attn_scores = [], []
for i in range(1, 4):
    with open(f'./output/log/{dir}/fold_{i}/dict_analysis.pickle', 'rb') as f:
        data = pickle.load(f)
    all_noise_amount.append(data['noise_amount'])
    all_attn_scores.append(data['attn_scores'])
        

all_noise_amount = np.concatenate(all_noise_amount)
all_attn_scores = np.concatenate(all_attn_scores)


before_tr, after_tr = [], []
for amount, weight in zip(all_noise_amount, all_attn_scores):
    if amount != 0:
        before_tr.append(weight[0][:amount+1].sum() * 100)
        after_tr.append(weight[0][amount+1:].sum() * 100)
np.mean(before_tr)
np.mean(after_tr)

self.attn_encoder

acc = [0.217, 0.156, 0.221]
earliness = [0.0126, 0.0121, 0.0139]
HM = [0.356, 0.27, 0.361]
np.mean(acc)
np.mean(earliness)
np.mean(HM)

np.mean([100, 75])
# ------------------------------------------------------------------------------ 
# 노이즈의 양에 따른 Basic과 attn의 퍼포먼스

import os
import utils
from dataset import CASAS_ADLMR, CASAS_RAW_NATURAL, CASAS_RAW_SEGMENTED, Dataloader


def group_results(logdir, model):
    concat_idx, concat_noise_amount, concat_attn_scores = [], [], []
    concat_true_y, concat_pred_y, concat_halt_prob, concat_halt_pnt = [], [], [], []
    for i in range(1, 4):
        with open(os.path.join(logdir, f'fold_{i}', 'dict_analysis.pickle'), 'rb') as f:
            dict_analysis = pickle.load(f)
            print(dict_analysis.keys())
        a = np.where(dict_analysis['all_yhat'] == -1, 0, 1)
        a = np.argmin(a, axis=1)
        halt_pnt = np.where(a == 0, dict_analysis['all_yhat'].shape[1], a) - 1
        halt_pnt = halt_pnt.reshape((-1, 1))
        pred_y = np.take_along_axis(dict_analysis['all_yhat'], halt_pnt, axis=1).flatten().astype(int)
        halt_prob = np.take_along_axis(dict_analysis['raw_probs'], halt_pnt, axis=1).flatten()
        concat_idx.append(dict_analysis['idx'])
        concat_noise_amount.append(dict_analysis['noise_amount'])
        if dict_analysis.get('attn_scores') is not None:
            concat_attn_scores.append(dict_analysis['attn_scores'])
        else:
            concat_attn_scores.append([])
        concat_true_y.append(dict_analysis['true_y'])
        concat_pred_y.append(pred_y)
        concat_halt_prob.append(halt_prob)
        concat_halt_pnt.append(halt_pnt)
        

    concat_idx = np.concatenate(concat_idx)
    concat_noise_amount = np.concatenate(concat_noise_amount)
    concat_attn_scores = np.concatenate(concat_attn_scores)
    concat_true_y = np.concatenate(concat_true_y)
    concat_pred_y = np.concatenate(concat_pred_y)
    if model == 'attn':
        concat_location = np.concatenate(concat_halt_pnt).flatten() + 1 + args.offset
    else:
        concat_location = np.concatenate(concat_halt_pnt).flatten() + 1 

    concat_lengths = data_natural.lengths[concat_idx]
    concat_location = np.where(concat_location > concat_lengths, concat_lengths, concat_location)

    idx_5 = np.where((concat_noise_amount >= 0) & (concat_noise_amount < 5))[0]
    idx_10 = np.where((concat_noise_amount >= 5) & (concat_noise_amount < 10))[0]
    idx_15 = np.where((concat_noise_amount >= 10) & (concat_noise_amount < 15))[0]
    idx_20 = np.where((concat_noise_amount >= 15) & (concat_noise_amount < 20))[0]

    acc_25 = np.where(concat_true_y[idx_5] == concat_pred_y[idx_5], 1, 0).mean()
    acc_50 = np.where(concat_true_y[idx_10] == concat_pred_y[idx_10], 1, 0).mean()
    acc_75 = np.where(concat_true_y[idx_15] == concat_pred_y[idx_15], 1, 0).mean()
    acc_100 = np.where(concat_true_y[idx_20] == concat_pred_y[idx_20], 1, 0).mean()

    earliness_25 = (concat_location[idx_5] / concat_lengths[idx_5]).mean()
    earliness_50 = (concat_location[idx_10] / concat_lengths[idx_10]).mean()
    earliness_75 = (concat_location[idx_15] / concat_lengths[idx_15]).mean()
    earliness_100 = (concat_location[idx_20] / concat_lengths[idx_20]).mean()

    correct = np.where(concat_true_y == concat_pred_y, 1, 0)
    
    acc_by_amount=[]
    for i in range(20):
        idx_amount = np.where(concat_noise_amount == i)[0]
        acc_by_amount.append(correct[idx_amount].mean())
    
    return [acc_25, acc_50, acc_75, acc_100], [earliness_25, earliness_50, earliness_75, earliness_100], concat_noise_amount, concat_attn_scores, correct, acc_by_amount




# 노이즈에 따른 구간 별 acc --------------------------------------------------------------------------------------
args = utils.create_parser()

args.with_other = False
args.balance = False
args.random_noise = True
args.except_all_other_events = False
data_natural = CASAS_RAW_NATURAL(args)


# activities = list(data_natural.label2idx.keys()) + ['All']
# 
# logdir = './output/log/220920-122104' # w/o PASS
# logdir = './output/log/220908-153132' # w/ PASS
# logdir = './output/log/220916-205235' # attn

basic_logdir_excOther = ['220921-225941'] # w/ PASS (Other events are excluded)
attn_logdir_excOther = ['220921-220448'] # attn  (Other events are excluded)

basic_logdir = ['220908-140057',
                '220908-143133',
                '220908-153132',
                '220908-170442',
                '220908-194622']

basic_rand_seed = ['220923-110959',
                    '220923-124519',
                    '220923-142429',
                    '220923-160043',
                    '220923-173808']


attn_logdir = ['220915-203726',
                '220915-210906',
                '220915-214253',
                '220915-222409',
                '220916-113309']

# attn_tr_points = ['220923-203510',
#                     '220923-210606',
#                     '220923-213757',
#                     '220923-221619',
#                     '220923-230213']

# attn_tr_points = ['220924-232921',
#                 '220925-000043',
#                 '220925-003150',
#                 '220925-010318',
#                 '220925-021823']

attn_cls_token = ['220926-151900',
                '220926-154933',
                '220926-162156',
                '220926-170742',
                '220926-183659']
attn_cls_token = ['220926-154933']

attn_noise_amount = ['220927-160021']

cnn_tuning = ['220928-222633',
            '220928-225317',
            '220928-232000',
            '220928-234818',
            '220929-001458',
            '220929-004212',
            '220929-010907',
            '220929-013533',
            '220929-020247',
            '220929-022925',
            '220929-025621',
            '220929-032248']
cnn_amount = ['220928-235539']

220928-223058
220928-225928
220928-232745
220928-235539


for corr in list_correct:
    corr.mean()


list_acc, list_earliness = [], []
list_noise_amount, list_attn_scores = [], []
list_correct, list_acc_by_amount = [], []
for logdir in cnn_amount:
    logdir = os.path.join('./output/log/', logdir)
    acc, earliness, noise_amount, attn_scores, correct, acc_by_amount = group_results(logdir, 'basic')
    list_acc.append(acc)
    list_earliness.append(earliness)
    list_noise_amount.append(noise_amount)
    list_attn_scores.append(attn_scores)
    list_correct.append(correct)
    list_acc_by_amount.append(acc_by_amount)
list_acc = np.array(list_acc)
list_earliness = np.array(list_earliness)
list_noise_amount = np.concatenate(list_noise_amount)
list_attn_scores = np.concatenate(list_attn_scores)
list_correct = np.concatenate(list_correct)
list_acc_by_amount = np.array(list_acc_by_amount)


basic_acc = np.mean(list_acc, axis=0)
# attn_acc = np.mean(list_acc, axis=0)


idx_5 = np.where((list_noise_amount >= 0) & (list_noise_amount < 5))[0]
idx_10 = np.where((list_noise_amount >= 5) & (list_noise_amount < 10))[0]
idx_15 = np.where((list_noise_amount >= 10) & (list_noise_amount < 15))[0]
idx_20 = np.where((list_noise_amount >= 15) & (list_noise_amount < 20))[0]


all_noise_amount = list_noise_amount[idx_20]
all_attn_scores = list_attn_scores[idx_20]

before_tr, after_tr = [], []
consecutive_three_before, consecutive_three_after = [], []
for amount, weight in zip(all_noise_amount, all_attn_scores):
    if amount != 0:
        if amount < 3 or amount > 17:
            continue
        before_tr.append(weight[0][:amount+1].mean() * 100)
        after_tr.append(weight[0][amount+1:].mean() * 100)
        consecutive_three_before.append(weight[0][amount] * 100)
        consecutive_three_after.append(weight[0][amount+1] * 100)
        # consecutive_three_before.append(weight[0][amount-2:amount+1].sum() * 100)
        # consecutive_three_after.append(weight[0][amount+1:amount+4].sum() * 100)
np.mean(before_tr)
np.mean(after_tr)
# np.mean(consecutive_three)
np.mean(consecutive_three_after) / np.mean(consecutive_three_before)
# np.mean(after_tr) / np.mean(before_tr)




idx_correct = np.where(list_correct == 1)[0]
list_noise_amount = list_noise_amount[idx_correct]
list_attn_scores = list_attn_scores[idx_correct]
list_correct = list_correct[idx_correct]

amount = 10
idx_10 = np.where(list_noise_amount == amount)[0]
attn_scores_10 = list_attn_scores[idx_10]
attn_scores_10[:,0,1:].mean(axis=0) * 100
# list_attn_scores[:,10,:].mean(axis=0) * 100


# noise amount와 earliness의 관계
best_HM = ['220929-201723',
            '220930-015421',
            '220930-083630',
            '220930-163443',
            '220930-005211',
            '220930-091900',
            '220930-004513',
            '220930-083124']


results = []
for logdir in best_HM:
    cv_amount_target, cv_amount_noise, cv_ratio = [], [], []
    for i in range(1, 4):
        path = os.path.join('./output/log/', logdir)
        with open(os.path.join(path, f'fold_{i}', 'dict_analysis.pickle'), 'rb') as f:
            dict_analysis = pickle.load(f)
            print(dict_analysis.keys())
        idx_correct = np.where(dict_analysis['true_y'] == dict_analysis['pred_y'])[0]
            
        cv_amount_target.append(np.mean(dict_analysis['locations'][idx_correct] - dict_analysis['noise_amount'][idx_correct]))
        cv_amount_noise.append(np.mean(dict_analysis['noise_amount'][idx_correct]))
        # ratio = (dict_analysis['locations'] - dict_analysis['noise_amount']) / dict_analysis['noise_amount']
        # cv_ratio.append(ratio.mean())
    results.append((np.mean(cv_amount_noise), np.mean(cv_amount_target)))



a.mean()


ls = []
a = np.array([1,2,3,4])
ls.append(a)
ls
tf.concat(ls, axis=1)
