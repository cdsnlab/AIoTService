# 중복값 찾기 
import pandas as pd

activity = ''  # empty
sensors, values, timestamps, activities, raw = [], [], [], [], []
with open('./dataset/cairo', 'rb') as features:
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
                    raw.append(des)
                    des = ''.join(f_info[4:])
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


idx = list(set(df_wo_raw[df_wo_raw.duplicated()].index) - set(df_w_raw[df_w_raw.duplicated()].index))

df_wo_raw.loc[idx]



# 순서 맞는지 확인
timestamps = []
with open('./dataset/milan', 'rb') as features:
    database = features.readlines()
    for i, line in enumerate(database):  # each line
        f_info = line.decode().split()  # find fields

        if '.' not in f_info[1]:
            f_info[1] = f_info[1] + '.000000'
        s = str(f_info[0]) + str(f_info[1])
        timestamps.append(time.mktime(datetime.strptime(s, "%Y-%m-%d%H:%M:%S.%f").timetuple()))

prev_t = 0
for i, t in enumerate(timestamps):
    if t < prev_t:
        print(i, t)
    prev_t = t
    
