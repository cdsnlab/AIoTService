config={
    'ws':           30, # window size
    'vs':            2,  # view size
    'threshold':    0.3, # dissimilarity score
    'interval':     10,  # true positive measure
}

feature_name={
    # every feature is defined in single window except prev
    0:"Hour",
    1:"Seconds",
    2:"Weekday",
    3:"Duration",
    4:"TimeFromPrev",
    5:"SensorPrev1",
    6:"SensorPrev2",
    7:"LastSensor",
    8:"LSLocation",
    9:"LMSLocation",
    10:"Complexity",
    11:"ActivityLevelChange",
    12:"Count",
    13:"ElapsedTime"
}