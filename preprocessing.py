# 중복값 찾기 
import pandas as pd

activity = ''  # empty
sensors, values, timestamps, activities, raw = [], [], [], [], []
with open('./dataset/kyoto11', 'rb') as features:
    database = features.readlines()
    for i, line in enumerate(database):  # each line
        f_info = line.decode().split()  # find fields
        try:
            if 'M' == f_info[2][0] or 'D' == f_info[2][0]:
                # choose only M D sensors, avoiding unexpected errors
                if '.' not in f_info[1]:
                    f_info[1] = f_info[1] + '.000000'
                s = str(f_info[0]) + str(f_info[1])
                timestamps.append(s)
                if f_info[3] == 'OPEN':
                    f_info[3] = 'ON'
                elif f_info[3] == 'CLOSE':
                    f_info[3] = 'OFF'
                sensors.append(f_info[2])
                values.append(f_info[3])

                if len(f_info) == 4:  # if activity does not exist
                    activities.append(activity)
                    raw.append('')
                    
                else:  # if activity exists
                    des = ''.join(f_info[4:])
                    raw.append(des)
                    if 'begin' in des:
                        activity = re.sub('begin', '', des)
                        activities.append(activity)
                    # if 'end' in des and activity == re.sub('end', '', des):
                    if 'end' in des:
                        activities.append(activity)
                        activity = ''
        except IndexError:
            print(i, line)
features.close()

df_wo_raw = pd.DataFrame({'sensors': sensors, 'values': values,
                        'timestamps': timestamps, 'activities': activities})
df_w_raw = pd.DataFrame({'sensors': sensors, 'values': values,
                        'timestamps': timestamps, 'activities': activities, "raw":raw})

idx = df_wo_raw[df_wo_raw.duplicated()].index
df_w_raw.loc[idx]

idx = list(set(df_wo_raw[df_wo_raw.duplicated()].index) - set(df_w_raw[df_w_raw.duplicated()].index))
df_wo_raw.loc[idx]


# 순서 맞는지 확인
timestamps = []
act = []
with open('./dataset/cairo', 'rb') as features:
    database = features.readlines()
    for i, line in enumerate(database):  # each line
        f_info = line.decode().split()  # find fields

        if '.' not in f_info[1]:
            f_info[1] = f_info[1] + '.000000'
        s = str(f_info[0]) + str(f_info[1])
        timestamps.append(time.mktime(datetime.strptime(s, "%Y-%m-%d%H:%M:%S.%f").timetuple()))
        act.append(f_info[-1])
len(timestamps)
len(database)

prev_t = 0
for i, t in enumerate(timestamps):
    if t < prev_t:
        print(i, t)
    prev_t = t
    
    
idx_sorted = np.argsort(timestamps)
sorted_act = np.array(act)[idx_sorted]
sorted_ts = np.array(timestamps)[idx_sorted]
database = sorted_act


prev_ts = 0
term = []
for ts in sorted_ts:
    if prev_ts == 0:
        prev_ts = ts
        continue
    term.append(ts - prev_ts)
    prev_ts = ts
np.argmax(term)
np.max(term) / 3600 / 24


database[535]
(np.max(timestamps) - np.min(timestamps)) / 3600 / 24

error1, error2 = [], []
big, end = 0, 0
with open('./dataset/milan', 'rb') as features:
    database = features.readlines()
    begin = False
    for i, line in enumerate(database):  # each line
        if i in []:
            continue
        f_info = line.decode().split()  # find fields
        # f_info = line
        if f_info[-1] == 'begin':
            big += 1
            if not begin:
                begin = True
            else:
                print(f'{i}:error')
                error1.append(i)
                continue
                # break
        elif f_info[-1] == 'end':
            end += 1
            if begin:
                begin = False
            else:
                print(f'{i}:error2')
                error2.append(i)
                continue
                # break
len(database)
len(set(database))
len(error1)
len(error2)
len(error1) / big

database[error[4]]
0.22606924643584522
0.13622559652928418



with open('./dataset/kyoto7_org', 'rb') as features:
    database = features.readlines()

with open('./dataset/data_kyoto7', 'rb') as features:
    database_dwn = features.readlines()


for i, (db1, db2) in enumerate(zip(database, database_dwn)):  # each line
    if db1 != db2:
        if i in [6629, 15222, 15375, 19258, 24645, 32378, 137788]:
            continue
        print(f"{i}: error")
        print(db1)
        print(db2)
        break

database_dwn[0] == database[0]



# filename = './dataset/kyoto11'
# activity = ''  # empty
# sensors, values, timestamps, activities = [], [], [], []
# with open(filename, 'rb') as features:
#     database = features.readlines()
#     for i, line in enumerate(database):  # each line
#         f_info = line.decode().split()  # find fields
#         try:
#             # if 'M' == f_info[2][0] or 'D' == f_info[2][0]:
#             # choose only M D sensors, avoiding unexpected errors
#             if '.' not in f_info[1]:
#                 f_info[1] = f_info[1] + '.000000'
#             s = str(f_info[0]) + str(f_info[1])
#             timestamps.append(int(time.mktime(datetime.strptime(s, "%Y-%m-%d%H:%M:%S.%f").timetuple())))
#             if f_info[3] == 'OPEN':
#                 f_info[3] = 'ON'
#             elif f_info[3] == 'CLOSE':
#                 f_info[3] = 'OFF'
#             sensors.append(f_info[2])
#             values.append(f_info[3])

#             if len(f_info) == 4:  # if activity does not exist
#                 activities.append(activity)
#             else:  # if activity exists
#                 des = ''.join(f_info[4:])
#                 if 'begin' in des:
#                     activity = re.sub('begin', '', des)
#                     activities.append(activity)
#                 # if 'end' in des and activity == re.sub('end', '', des):
#                 if 'end' in des:
#                     activities.append(activity)
#                     activity = ''
#             if f_info[2][0] not in ['M', 'D']:
#                 del sensors[-1]
#                 del values[-1]
#                 del timestamps[-1]
#                 del activities[-1]
#         except IndexError:
#             print(i, line)
# features.close()
# set(i[0] for i in set(sensors))



# 센서간 correlation -------------------------------------------------------------------------------------------
args = utils.create_parser()
args.with_other = False
args.balance = False
args.random_noise = True
args.except_all_other_events = False
data_natural = CASAS_RAW_NATURAL(args)

data_natural.state_matrix.shape


def calc_duration_corr(duration):
    list_duration_corr = []
    for i in range(len(duration)):
        diff = np.abs(duration - duration[i])
        pivot = np.broadcast_to(duration[i], (data_natural.N_FEATURES))
        duration_max = np.max([duration, pivot], axis=0) 
        duration_max[np.where(duration_max==0)[0]] = 0.0001
        duration_corr = 1 - (diff / duration_max)
        list_duration_corr.append(duration_corr)
    return np.array(list_duration_corr)
    
duration = np.zeros((data_natural.N_FEATURES))
flag = np.zeros((data_natural.N_FEATURES))
duration_corr = []
for i, state in enumerate(data_natural.state_matrix):
    new_ON_idx = np.where((state==1) & (flag == 0))[0]
    new_OFF_idx = np.where((state==0) & (flag == 1))[0]
    # new_OFF_idx = np.where(((state==0) & (flag == 1)) | (count>=10))[0]
    
    flag[new_ON_idx] = 1
    activated_idx = np.where(flag==1)[0]
    duration[activated_idx] += 1
    
    duration_corr.append(calc_duration_corr(duration))
    
    flag[new_OFF_idx] = 0
    duration[new_OFF_idx] = 0
    if i % 100 == 0:
        print(i)
    
# ------------------------------------------------------------------------------------------------
# Lapras dataset preprocessing
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data_path = "./dataset/Lapras_Raw"
episodes = []
X, X_file = [], []
flag = False
for wd in glob.glob(f"{data_path}/*"):
    activity = wd.split("/")[-1]
    # print(activity)
    filelist = sorted(glob.glob(f"{wd}/*.csv"))
    for file in filelist:
        df = pd.read_csv(file, header=None)
        for x, x_f in zip(X, X_file):
            # if (len(df) == len(x)) and (np.sum(x == df.values) == (len(df) * 3)):
            if (len(df) == len(x)) and df[2][0] == x[0][2]:
                flag = True
                print(x_f, file)
                break
        if flag:
            flag = False
            continue   
        else:
            X.append(df.values)
            X_file.append(file)
        df = df.loc[df[0].str.contains(r"Present|Brightness|LightGroup|ProjectorPower|Screen|Seat1|Seat2|Seat3|Seat4|Seat5|Seat6|Sound|WhiteboardUsed", case=False)]
        df = df.loc[~df[0].str.contains(r"_Brightness|TurnOffLight|TurnOnLight", case=False)]
        if activity in ['Presentation', 'Discussion']:
            idx = np.where(df[0].str.contains(r"ProjectorPower", case=False).to_numpy())[0]
            if (len(idx) == 0) or ("On" not in df.iloc[idx][1].to_list()):
                start_time = df[2].to_list()[0]
                new_row = pd.DataFrame([['ProjectorPower', 'On', start_time]], columns = df.columns)
                df = pd.concat([new_row, df], ignore_index = True)
        df.to_csv(f'./dataset/Lapras_rm_ctxt/{activity}/{file.split("/")[-1]}', header=False, index=False)



# scaler = StandardScaler()
scaler = MinMaxScaler()

data_path = "./dataset/Lapras_rm_ctxt"
episodes = []
normalize_context = ['Brightness', 'SoundC', 'SoundLeft', 'SoundRight', 'Present']
for wd in glob.glob(f"{data_path}/*"):
    activity = wd.split("/")[-1]
    # print(activity)
    filelist = sorted(glob.glob(f"{wd}/*.csv"))
    for file in filelist:
        df = pd.read_csv(file, header=None)
        for c in normalize_context:
            idx = df[df[0]==c][1].index
            if len(idx) == 0:
                continue
            values = df[df[0]==c][1].to_numpy().astype(float).reshape(-1, 1)
            
            scaler.fit(values)
            scaled = scaler.transform(values)
            df.iloc[idx, 1] = scaled.flatten()
        # df.to_csv(f'./dataset/Lapras_standardized/{activity}/{file.split("/")[-1]}', header=False, index=False)
        df.to_csv(f'./dataset/Lapras_normalized/{activity}/{file.split("/")[-1]}', header=False, index=False)

