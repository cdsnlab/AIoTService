config={
    'ws':           30, # window size
    'vs':            2,  # view size
    'threshold':    0.1, # dissimilarity score
    'interval':     10,  # true positive measure
}

feature_name=[
    # every feature is defined in single window except prev
    "HourOfDay",                        #0  
    "SecondOfDay",                      #1
    "DayOfWeek",                        #2
    "Duration",                         #3
    "ElapsedTimeFromLastEvent",         #4
    "ActivityLevelChange",              #5  A'
    "DominantSensorPrev1W",             #6
    "DominantSensorPrev2W",             #7
    "DominantSensorCurrentW",           #8  A'
    "FirstSensor",                      #9
    "LastSensor",                       #10
    "LastSensorLocation",               #11
    "LastMotionSensorLocation",         #12
    "DominantSensorLocation",           #13 A'
    "DataComplexity",                   #14
    "DistinctSensors",                  #15 A'
    "Count",                            #16
    "ElapsedTime"                       #17
]

exclude_list = {
    'A': [5, 8, 13, 15],
    'B': [4, 6, 8, 12, 16]
}