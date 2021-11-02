import numpy as np
import glob

from .readText import read_hh, read_adlmr, create_episodes, time_correction
from .info.testbed import activityfiles_new as files
# from .featureExtraction import 

from .info.hh import baseline_activities

from itertools import permutations



def dataLoader(dataset, option=None):
    if dataset=="hh101":
        if option:
            return load_hh101(remove_other=True)
        else:
            return load_hh101()
    if dataset=="adlmr":
        return load_adlmr()
    if dataset=="testbed":
        return load_testbed()
    
    raise ValueError("Wrong dataset.")

def load_hh101(remove_other=False):
    # Load RAW
    with open("./dataset/hh/hh101/ann.txt", "rb") as f:
        rawevents = read_hh(f.readlines())

    # Subtract all EXCEPT Motion and Door
    mdevents = [event for event in rawevents if event[0][0] in ['M','D']]

    transitions=[]
    for i in range(1, len(mdevents)):
        if mdevents[i][-1]!=mdevents[i-1][-1]:
            transitions.append(i)
    
    activities=[]
    previdx = 0
    for i in range(len(transitions)):
        activity = np.array(mdevents[previdx:transitions[i]])
        assert len(set(activity[:,-1]))<=1
        for j in range(len(activity)):
            activity[j,-1] = baseline_activities[activity[j,-1]]
        activities.append(activity)
        previdx = transitions[i]
        if i==len(transitions)-1:
            activity = np.array(mdevents[previdx:])
            assert len(set(activity[:,-1]))<=1
            for j in range(len(activity)):
                activity[j,-1] = baseline_activities[activity[j,-1]]
            activities.append(activity)

    if remove_other:
        activities = [activity for activity in activities if activity[0][-1]!="Other"]
        print("Remove Other.")

    indices=[]
    prevright = 0
    for i in range(1, len(activities)-1): # SKIP THE FOREMOST ONE (OUTLIER)
        left, right = activities[i][0,-1], activities[i+1][0,-1]
        if left==right or right=="Sleep":
            continue
        if i == prevright:
            if len(activities[i+1])>5: # Heuristic: Exclude too short event (less than 5 events)
                indices.append(i+1)
        else:
            if len(activities[i])>5 and len(activities[i+1])>5:
                indices.append(i); indices.append(i+1)
        prevright = i+1
    
    assert len(set(indices))==len(indices)

    episodes = []
    transitions = []
    labels = []
    for idx in range(len(indices)-1):
        left, right = activities[indices[idx]], activities[indices[idx+1]]
        assert len(set(left[:,-1]))==1
        assert len(set(right[:,-1]))==1
        if left[0,-1]==right[0,-1]:
            continue
        
        transitions.append(len(left))
        episodes.append(np.concatenate((left, right)))
        labels.append("{}-{}".format(left[0,-1], right[0,-1]))
    
    episodes = [time_correction(eps, transitions[i]) for i, eps in enumerate(episodes)]

    print("""hh101 Data Load Completed.
    Number of Activity Pairs: {}
    """.format(len(episodes)))
    
    return episodes, transitions, labels

def load_adlmr():
    with open("./dataset/adlmr/annotated", 'rb') as f:
        rawevents = read_adlmr(f.readlines())

    # Subtract all EXCEPT Motion and Door
    mdevents = [event for event in rawevents if event[0][0] in ['M','D']]

    transitions=[]
    for i in range(1, len(mdevents)):
        if mdevents[i][-1]!=mdevents[i-1][-1]:
            transitions.append(i)
    
    activities=[]
    previdx = 0
    for i in range(len(transitions)):
        activity = np.array(mdevents[previdx:transitions[i]])
        assert len(set(activity[:,-1]))<=1
        activities.append(activity)
        previdx = transitions[i]
        if i==len(transitions)-1:
            activity = np.array(mdevents[previdx:])
            assert len(set(activity[:,-1]))<=1
            activities.append(activity)


    series_number = 10
    
    episodes = []
    transitions = []
    labels = []
    for i in range(len(activities)-1):
        l, r = activities[i:i+2]
        merge_activity = np.concatenate([l, r])
        merge_activity = time_correction(merge_activity, len(l))
        episodes.append(merge_activity)
        transitions.append(len(l))
        
        assert len(set(l[:,-1]))*len(set(r[:,-1]))==1
        labels.append(f"{l[0,-1]}-{r[0,-1]}")

    return episodes, transitions, labels

def load_testbed():

    testbed_directory = "./dataset/testbed/npy/seminar/MO/B"

    tasks = sorted(set([fd.split("/")[-1].split(".")[0][:-2] 
        for fd in glob.glob(f"{testbed_directory}/*.npy")]))

    print(f"tasks: {tasks}")

    task_dict = {task: [] for task in tasks}

    for filedir in sorted(glob.glob(f"{testbed_directory}/*.npy")):
        activity = np.load(filedir)
        filename = filedir.split("/")[-1].split(".")[0]
        label, number = filename[:-2], filename[-2:]
        
        task_dict[label].append((number, activity))

    assert len(tasks)==len(task_dict.keys())

    episodes = []
    transitions = []
    labels = []

    perm = list(permutations(tasks, 2))    

    for pair in perm:
        l, r = pair

        assert l!=r

        lan, la = task_dict[l].pop()
        ran, ra = task_dict[r].pop()

        merge_activity = np.concatenate([la, ra])
        merge_activity = time_correction(merge_activity, len(la))

        episodes.append(merge_activity)
        transitions.append(len(la))
        labels.append(f"{l}{lan}-{r}{ran}")

    return episodes, transitions, labels