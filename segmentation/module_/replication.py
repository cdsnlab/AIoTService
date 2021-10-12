import numpy as np
from matplotlib import pyplot as plt
from glob import glob

from .readText import read_hh
from .featureExtraction import feature_extraction
from .info.hh import baseline_activities
from .info.config import feature_name, exclude_list
from .changePointDetection import change_point_detection
from .info.testbed import activityfiles_new as files
from .readText import create_episodes, time_correction


def replication(dataset, metric):
    """
        1. Data Collection
        2. Data Preprocessing
        3. Sliding Window and Feature Extraction
        4. Change Point Detection
    """
    
    fig = plt.figure(figsize=(10, 1))

    total = 0
    if dataset=="check":
        testbed_directory = "./dataset/testbed/npy/seminar/MS"
        task_dict = {i: [np.load("{}/{}".format(testbed_directory, name)) for name in v] for i, v in enumerate(files.values())}
        label_dict = {i: k for i, k in enumerate(files.keys())}

        episodes, transitions, labels = create_episodes(task_dict, label_dict)
        episodes = [time_correction(eps, transitions[i]) for i, eps in enumerate(episodes)]

        # result = 0
        for i in range(len(episodes)):
            episode, transition, label = episodes[i], transitions[i], labels[i]

            scores = np.load("./replication/testbed/{}_{}.npy".format(label, metric))

            for j in range(50, transition-50):
                if scores[j]>0.5:
                    total+=1   

        return total

    if dataset=="testbed":
        # Data Collection - testbed
        testbed_directory = "./dataset/testbed/npy/seminar/MS"
        task_dict = {i: [np.load("{}/{}".format(testbed_directory, name)) for name in v] for i, v in enumerate(files.values())}
        label_dict = {i: k for i, k in enumerate(files.keys())}

        episodes, transitions, labels = create_episodes(task_dict, label_dict)
        episodes = [time_correction(eps, transitions[i]) for i, eps in enumerate(episodes)]

        result = 0
        for i in range(len(episodes)):
            episode, transition, label = episodes[i], transitions[i], labels[i]

            sensor_list = sorted(set(episode[:,0]))

            features = np.array(feature_extraction(episode, dataset, sensor_list))

            assert len(features)==len(episode)

            scores = np.array(change_point_detection(features, metric))

            assert len(scores)==len(features)

            ax = fig.add_subplot(1, 1, 1)

            np.save("replication/{}/{}_{}.npy".format(dataset, label, metric), scores)
            if "uLSIF" in metric:
                scores[scores<0] = 0
            plt.plot(range(len(scores)), scores)
            plt.axvline(x=transition, linestyle=":", color="g")
            plt.title(label)
            plt.savefig("./replication/{}/{}_{}.png".format(dataset, label, metric))
            plt.clf()
            
            flag=False
            for j in range(transition-50, transition+50):
                if scores[j]>0.5:
                    flag=True
                    break
            if flag:
                result+=1

            print("{}/{} {} {}".format(i, len(episodes)-1, label, flag))

        print(result, len(episodes))

    if dataset=="hh101":
        # Data Collection - hh101
        with open("dataset/hh/hh101/ann.txt", "rb") as f:
            rawdata = f.readlines()
        events = read_hh(rawdata)
        print("Load Raw Data: {} events".format(len(events)))
    
        # Filter sensor
        events_1 = [event for event in events if event[0][0] in ['M','D']]
        print("Use Motion and Door Sensors: {} events".format(len(events_1)))

        # Transitions Points
        transitions=[]
        for i in range(1, len(events_1)):
            if events_1[i][-1]!=events_1[i-1][-1]:
                transitions.append(i)
        print("Raw Transitions: {}".format(len(transitions)))
        
        # Create Sequential List of Activity
        activities=[]
        previdx = 0
        for i in range(len(transitions)):
            activity = np.array(events_1[previdx:transitions[i]])
            assert len(set(activity[:,-1]))<=1
            for j in range(len(activity)):
                activity[j,-1] = baseline_activities[activity[j,-1]]
            activities.append(activity)
            previdx = transitions[i]
            if i==len(transitions)-1:
                activity = np.array(events_1[previdx:])
                assert len(set(activity[:,-1]))<=1
                for j in range(len(activity)):
                    activity[j,-1] = baseline_activities[activity[j,-1]]
                activities.append(activity)
        print("Create Raw Activity List.")

        # Preprocess Activities based on Transition Distribution
        l=[]
        prevright = 0
        for i in range(1, len(activities)-1): # SKIP THE FOREMOST ONE (OUTLIER)
            left, right = activities[i][0,-1], activities[i+1][0,-1]
            if left==right or right=="Sleep":
                continue
            if i == prevright:
                if len(activities[i+1])>5:
                    l.append(i+1)
            else:
                if len(activities[i])>5 and len(activities[i+1])>5:
                    l.append(i); l.append(i+1)
            prevright = i+1
        
        assert len(set(l))==len(l)
        print("Construct Dataset based on Distribution: {} activities".format(len(l)))

        ppevents = np.concatenate([activities[i] for i in l])
        pptransitions = []
        previdx = 0
        for i in range(len(l)-1):
            pptransitions.append(previdx+len(activities[l[i]]))
            previdx+=len(activities[l[i]])

        sensor_list = sorted(set(ppevents[:,0]))
        print("""Preprocessed events: {}, 
        transitions: {},
        the number of sensors: {}""".format(len(ppevents), len(pptransitions), len(sensor_list)))

        np.save("replication/hh101/ppevents.npy", ppevents)
        np.save("replication/hh101/pptransitions.npy", pptransitions)

        features = np.array(feature_extraction(ppevents, "hh101", sensor_list))

        assert len(features)==len(ppevents)

        ppfeatures = []
        for i in range(features.shape[1]):
            if i not in exclude_list["A"]:
                ppfeatures.append(features[:,i].reshape(-1,1))
        ppfeatures = np.concatenate(ppfeatures, axis=1)

        assert ppfeatures.shape[1]==len(feature_name)-len(exclude_list["A"])-2+2*len(sensor_list)
        print("Feature shape: {}".format(ppfeatures.shape))

        np.save("replication/hh101/ppfeatures.npy", ppfeatures)

        scores = np.array(change_point_detection(ppfeatures, metric))

        assert len(scores)==len(ppfeatures)

        np.save("replication/hh101/scores_{}.npy".format(metric), scores)

    return