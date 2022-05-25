import dataclasses
from calendar import c
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
example_img.save('./analysis/heatmap/heatmap_milan.png')


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