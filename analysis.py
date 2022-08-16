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

experiment_id = "CoVG6yTnQoC3da66UYozJQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df
# dfw = experiment.get_scalars(pivot=True) 
# dfw

df1 = df.loc[df['run'].str.contains(r"test")]
df1[df1["step"] == 49].groupby("tag").mean()



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
    lam = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0]
    df_list = []
    for i, experiment_id in enumerate(experiment_ids):
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        df = experiment.get_scalars()
        df = df.loc[df['run'].str.contains(r"test")]
        df = df[df["step"] == 49]
        df = df[['tag', 'value']]
        df["lambda"] = lam[i]
        df_list.append(df)
    df_concat = pd.concat(df_list)
    df_concat = df_concat.pivot(index='lambda', columns='tag', values='value')
    df_concat = df_concat.sort_values(by="lambda", ascending = False)
    df_concat.index = df_concat.index.astype(str)
    df_concat = df_concat[['whole_accuracy', 'whole_count_mean', 'whole_earliness', 'whole_location_mean']]
    df_concat.rename(columns={"whole_accuracy": "Accuracy", 
                            "whole_count_mean": "# used event", 
                            "whole_earliness": "Earliness", 
                            "whole_location_mean": "Waiting seconds"}, 
                    inplace=True)
    df_concat['Earliness'] = df_concat['Earliness'] * 100
    df_concat['Accuracy'] = df_concat['Accuracy'] * 100
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

    return df_concat[['Accuracy', 'Earliness', 'Waiting seconds', '# used event']]

experiment_ids_wo_other = ['A7oqlUEuRteRhbJyYRAZfg','gzZLlGroTAeSZWPt26V0mw','2eZoNUWZQOSyUwxcm31x1A','sMb8DC93RYmXMrZJPzTpiA','L30wvOFYSb2JvRtqf47aJA','0T4zC5BCReKeAM4Mvb9cKg','v0oGZndxRbOIlbKUJ2YUXw','2KOdOINrSveCn7WbYJ9yFQ','QAd5ilRXT7qVQAV2tpuR0Q','XFpbnJbhRp2UXOEkKtvTbA',]
experiment_ids_w_other = ['vvfpZu4xQAmpQ6wDHxTBOw','YivqHn6gQPi4uQ8ZWk6sDQ','QU1PmCx5Qem6pPpj1D04vA','1XszRYfaQRCXubrB2Pjaig','onNHRy80SDuYWsAYoKvjLA','pd4KvqPPTdOeu62kS3rqqg','fgmrdXs9TDOAlfQFJEiZyQ','2PRN4seSREqJ2Saf01jKtg','aPUKI1a8R7eWLfMasJ3hzA','71qmw4bQR0aFlBC8yIRHXw']

df_wo_other = acc_by_lam(experiment_ids_wo_other, 'wo_other')
df_w_other = acc_by_lam(experiment_ids_w_other, 'w_other')


# -----------------------------------------------------------------------------------------------------------------
# Data characteristics
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import *

args.dataset = "milan"
args.with_other = True
data = CASAS_RAW_NATURAL(args)
np.max(data.lengths)

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

df = pd.DataFrame({"class":data.Y, "duration": data.lengths})
df['class'] = list(map(lambda x: data.idx2label[x], df['class']))
# df.groupby("class").mean().to_numpy()
df = df[df["duration"] > 0]
df["duration"].describe()
df["duration"].quantile(0.88)
df = df[df["duration"] < 2000]
df["duration"] = df["duration"] / 60
df["duration"].describe()

# 전체 데이터에 대한 히스토그램
plt.hist(df["duration"], bins=50)
plt.title("Histogram of activity duration")
plt.xlabel("Duration (min)")
plt.ylabel("Frequency")
plt.show()
plt.savefig("./analysis/hist_duration.png")
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
plt.savefig("./analysis/boxplot.png")
plt.clf()


# 클래스 별 평균 길이
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
args.with_other = False
args.balance = True
data = CASAS_RAW_NATURAL(args)
logdir = './output/log/220802-212138'
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
np.where(dict_analysis['true_y'] == pred_y, 1, 0).mean

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
for subseq_len in subsequence_lengths:
    # calc_duration(data=data, target_class='Bathing', subseq_len=subseq_len)
    calc_duration(data=data, target_class='Bed_to_toilet', subseq_len=subseq_len)


# 발생 시간대
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
logdir = './output/log/220812-151545'
# concat_entropy_wrong, concat_entropy_correct = [], []
concat_entropy_all = {a: [] for a in activities}
concat_entropy_wrong = {a: [] for a in activities}
concat_entropy_correct = {a: [] for a in activities}
concat_true_y, concat_pred_y, concat_halt_prob = [], [], []
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
np.where(concat_true_y == concat_pred_y, 1, 0).mean()


dir = os.path.join(logdir, f'fold_{1}', 'confusion_matrix_real.png')
utils.plot_confusion_matrix(concat_true_y, concat_pred_y, dir, target_names=list(data_natural.idx2label.values()))

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

