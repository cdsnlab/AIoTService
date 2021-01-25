import pandas as pd
import numpy as np
import math
from datetime import datetime
from densratio import densratio
from scipy.stats import multivariate_normal
import pickle
from matplotlib import pyplot as plt
import argparse
import sys
'''
# 2018-10-05 {'Chatting': [25], 'Discussion': [49], 'GroupStudy': [34], 'NULL': [63], 'Presentation': [120]}
# 2018-10-02 {'Chatting': [119], 'Discussion': [45, 46], 'GroupStudy': [31], 'NULL': [78, 94], 'Presentation': [117, 118]}
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-dre', type=bool, default=False, help='density ratio estimation')
    parser.add_argument('-plt', type=bool, default=True, help='plot ratio')
    parser.add_argument('-vw', type=int, default=30, help='the size of view')
    parser.add_argument('-f', type=str, default="", help='dissimilarity file name')
    parser.add_argument('-a', type=float, default=0.0, help='alpha value')
    args=parser.parse_args()
    print(args)

    if args.f!="":
        with open('./dissimilarity/{}.txt'.format(args.f), 'rb') as fp:
            dissimilarity=pickle.load(fp)

        plt.plot(range(1, len(dissimilarity)+1), dissimilarity)
        # for item in transition_section:
        #     plt.axvline(x=item, linestyle=':', color='r')
        plt.savefig("./img/{}_nogt.png".format(args.f))
        sys.exit(0)

    oct5Date={
        'Chatting': 25,
        # 'Discussion': 49,
        'GroupStudy': 34, 
        # 'NULL': 63, 
        'Presentation': 120
    }

    oct2Date={
        'Chatting': 119,
        'Discussion': 46,
        'GroupStudy': 31, 
        'NULL': 94, 
        'Presentation': 118
    }

    feature_names=[
        'dayofweek',
        'hourofday',
        'secpastmidnight',
        'duration',
        'mostrecentsid',
        'activitylevelchange',
        'prevmostfreqsid',
        'countofevents',
        'mostfreqsid',
        'elapsedtimeofevents',
    ]

    sensor_id={}
    sensor_init={}
    total_sensor_set=set()
    total_df=None
    target_date=oct5Date
    sid_offset=1
    view_size=args.vw
    transition_section=[]
    amount_of_rows=0

    # concatenate streams and find unique set of all sensors included
    for i, item in enumerate(target_date.keys()):
        df=pd.read_csv("./data/{}{}.csv".format(item, target_date[item]), header=None)
        df.columns=['name','value','timestamp']
        amount_of_rows+=len(df.index)
        transition_section.append(math.ceil(amount_of_rows/view_size))
        if i==0:
            total_df=df
        else:
            total_df=pd.concat([total_df, df])
        total_sensor_set=total_sensor_set.union(set(df.name.unique()))
    total_sensor_set=sorted(total_sensor_set)
    unique_sensor_length=len(total_sensor_set)
    print(total_sensor_set)
    print("The number of unique sensors: {}".format(unique_sensor_length))
    sys.exit(0)
    # initialize dict for the definition of sid and its count
    for i, item in enumerate(total_sensor_set):
        sensor_init[item]=0
        sensor_id[item]=sid_offset+i

    # remove reminder to reshape dataframe
    total_df=total_df.to_numpy()
    total_df=total_df[:-1*(len(total_df)%view_size)]
    total_df=np.reshape(total_df, (-1, view_size, 3))

    features=[]
    total_duration=0
    prev_most_freq_sid=0
    
    # cnt=0
    for i, window in enumerate(total_df):
        window_feature=[]

        begin_time=window[0][-1]
        date=str(datetime.fromtimestamp(begin_time/1000))
        pd_ts=pd.Timestamp(begin_time/1000, unit='s', tz='Asia/Seoul')

        ## (1) time features
        # 1-1. day of week
        window_feature.append(pd_ts.dayofweek)
        # 1-2. hour of day
        window_feature.append(pd_ts.hour)
        # 1-3. seconds past midnight
        spm=pd_ts.hour*3600+pd_ts.minute*60+pd_ts.second
        window_feature.append(spm)

        ## (2) window features
        duration=abs(window[0][-1]-window[-1][-1])
        # because of the concatenation of streams in different time.
        if duration>30000:
            avg_duration=int(total_duration/(i+1))
            duration=avg_duration
        # if duration<0:
        #     cnt+=1
        total_duration+=duration
        # 2-1. window duration
        window_feature.append(duration/1000)

        sensor_count_dict=sensor_init.copy()
        sensor_time_last=sensor_init.copy()
        sensor_first_half=None
        most_recent_sid=0
        for j, item in enumerate(window):
            if j==round(view_size/2)-1:
                sensor_first_half=item[-1]
            sensor_count_dict[item[0]]+=1
            sensor_time_last[item[0]]=item[-1]

            # 2-4. most recent sensor
            if j==len(window)-1:
                most_recent_sid=sensor_id[item[0]]
                window_feature.append(most_recent_sid)

        first_half_duration=sensor_first_half-begin_time
        # 2-2. activity level change 
        # (the ratio btw time duration of 1st half and total duration)
        window_feature.append(first_half_duration/duration)

        ## (3) sensor features
        # 3-3. prev most freq sensor
        window_feature.append(prev_most_freq_sid)
        max_cnt=-1
        most_freq_sid=0

        for item in sensor_id.keys():
            # 3-1. the number of occurence of each sensor
            window_feature.append(sensor_count_dict[item])
            if max_cnt<sensor_count_dict[item]:
                max_cnt=sensor_count_dict[item]
                most_freq_sid=sensor_id[item]

        # 2-3. most frequently activated sensor in the window
        window_feature.append(most_freq_sid)
        prev_most_freq_sid=most_freq_sid

        for item in sensor_time_last.keys():
            if sensor_time_last[item]==0:
                window_feature.append(-1)
                continue
            elapsed_time=abs(begin_time+duration-sensor_time_last[item])
            # because of the concatenation of streams in different time.
            if elapsed_time>duration:
                elapsed_time=0.5*duration
            # 3-2. elapsed time from last fired to end of window
            window_feature.append(elapsed_time/1000)
        
        features.append(np.array(window_feature))
    # print(features[0])
    dissimilarity=[]

    
    print("The number of views: {}".format(len(features)))

    if args.dre:
        np.random.seed(1)
        for i in range(len(features)-1):
            print('{} - {}'.format(i, i+1))
            densratio_obj=densratio(    features[i],
                                        features[i+1],
                                        alpha=args.a,
                                        sigma_range=[0.1, 0.3, 0.5, 0.7, 1], 
                                        lambda_range=[0.01, 0.02, 0.03, 0.04, 0.05],
                                        verbose=False)
            dissimilarity.append(densratio_obj.alpha_PE)

        with open('./dissimilarity/rulsif_oct2.txt', 'wb') as fp:
            pickle.dump(dissimilarity, fp)

    if args.plt:
        with open('./dissimilarity/rulsif_oct2.txt', 'rb') as fp:
            dissimilarity=pickle.load(fp)

        plt.plot(range(1, len(dissimilarity)+1), dissimilarity)
        # for item in transition_section:
        #     plt.axvline(x=item, linestyle=':', color='r')
        plt.savefig("./img/rulsif_oct2_nogt.png")