config={
    'ws':           30, # window size
    'vs':            2,  # view size
    'threshold':    0.1, # dissimilarity score
    'interval':     10,  # true positive measure
}

feature_name=[
    # every feature is defined in single window except prev
    "Hour",                     #0
    "Sec",                      #1
    "Weekday",                  #2
    "Duration",                 #3
    "TimeBtwPrevEvent",         #4
    "DominantSensorPrev1",      #5
    "DominantSensorPrev2",      #6
    "DominantSensorCurrent",    #7
    "FirstSensor",              #8
    "LastSensor",               #9
    "DominantSensorLocation",    #10
    "LastSensorLocation",       #11
    "LastMotionLocation",       #12
    "DataComplexity",           #13
    "ActivityLevelChange",      #14
    "MotionTransition",         #15
    "DistinctSensors",          #16
    "Count",                    #17
    "ElapsedTime"               #18
]

exclude_list = {
    'A': [7, 10, 14, 15, 16],
    'B': [4, 6, 8, 12, 16]
}