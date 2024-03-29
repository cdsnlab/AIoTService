import numpy as np

activityfiles_old={
    'Chatting': [
"180110_150824_C.npy",
'180212_145745_C.npy',
'180406_154307_C.npy',
'180117_154826_C.npy',
'180531_184301_C.npy'
    ],
    'TechnicalDiscussion': [
'180711_095832_D.npy',
'180405_165701_D.npy',
'180912_155838_D.npy',
'180418_130100_D.npy',
'180109_155900_D.npy'
    ],
    'GroupStudy': [
'180615_102702_G.npy',
'180419_144100_G.npy',
'180529_130555_G.npy',
'180904_085140_G.npy',
'180813_130545_G.npy'
    ],
    'Seminar': [
'180724_100001_P.npy',
'180604_095300_P.npy',
'180521_095902_P.npy',
'180529_095851_P.npy',
'180724_142401_P.npy',
    ]
}

activityfiles_new={
    'Chatting': [
"Chatting45.npy",
"Chatting53.npy",
"Chatting64.npy",
"Chatting72.npy",
"Chatting82.npy"
    ],
    'TechnicalDiscussion': [
'Discussion1.npy',
'Discussion20.npy',
'Discussion23.npy',
'Discussion24.npy',
'Discussion25.npy'
    ],
    'GroupStudy': [
'GroupStudy10.npy',
'GroupStudy12.npy',
'GroupStudy14.npy',
'GroupStudy15.npy',
'GroupStudy16.npy'
    ],
    'Seminar': [
'Presentation100.npy',
'Presentation105.npy',
'Presentation110.npy',
'Presentation111.npy',
'Presentation112.npy'
    ]
}



testbed_location={
    'AC0': (10,13), #
    'AC1': (10,13),  #
    'PJ': (10,13),  #
    'LT': (10,13),  # -
    'DR': (16,0),   #
    'PE': (10,13),  # -
    'SC': (10,13),  #

    'MT0': (10,13),
    'MT1': (10,13),
    'MT2': (10,13),
    'MT3': (10,13),
    'MT4': (10,13),
    'MT5': (10,13),
    'MT6': (10,13),
    'MT8': (10,13),
    'MT9': (10,13),

    'SE1A': (10,13),
    'SE1B': (10,13),
    'SE2A': (10,13),
    'SE2B': (10,13),
    'SE3A': (10,13),
    'SE3B': (10,13),
    'SE4A': (10,13),
    'SE4B': (10,13),
    'SE5A': (10,13),
    'SE5B': (10,13),
    'SE6A': (10,13),
    'SE6B': (10,13),
    'SE7B': (10,13),

    'SOC0': (10,13),
    'SOC1': (10,13),
    'SOC2': (10,13),
    'SOL':  (10,13),
    'SOR':  (10,13),
    'SOWA0': (10,13),
    'SOWA1': (10,13),
    'SOWA2': (10,13),
    'SOWI0': (10,13),
    'SOWI1': (10,13),
    'SOWI2': (10,13),
}

seminar_location={
    # 'Aircon-Door': (10,13),
    # 'Aircon-Screen': (10,13),
    'Door': (10,13),
    'Light': (10,13),
    'Projector': (10,13),
    # 'Seat1A': (10,13),
    # 'Seat1B': (10,13),
    # 'Seat2A': (10,13),
    # 'Seat2B': (10,13),
    # 'Seat3A': (10,13),
    # 'Seat3B': (10,13),
    # 'Seat4A': (10,13),
    # 'Seat4B': (10,13),
    # 'Seat5A': (10,13),
    # 'Seat5B': (10,13),
    # 'Seat6A': (10,13),
    # 'Seat6B': (10,13),
    'Motion1': (10,13),
    'Motion2': (10,13),
    'Motion3': (10,13),
    'Motion4': (10,13),
    'Motion5': (10,13),
    'Motion6': (10,13),
    # 'Motion7': (10,13),
    'Motion8': (10,13),
    'Motion9': (10,13),
}

motion_euclidean_distance = \
    [
        [0, 1.035, 1.761, 1.739, 1.816, 1.419, 1.362, 1.118],
        [1.035, 0, 1.693, 1.589, 1.770, 1.650, 1.218, 1.113],
        [1.761, 1.693, 0, 1.350, 1.012, 1.537, 1.513, 1.621],
        [1.739, 1.589, 1.350, 0, 1.066, 1.248, 1.717, 1.754],
        [1.816, 1.770, 1.012, 1.066, 0, 1.381, 1.567, 1.745],
        [1.419, 1.650, 1.537, 1.248, 1.381, 0, 1.759, 1.594],
        [1.362, 1.218, 1.513, 1.717, 1.567, 1.759, 0, 1.085],
        [1.118, 1.113, 1.621, 1.754, 1.745, 1.594, 1.085, 0]
    ]


motion_cosine_distance = \
    [
        [0, 0.535, 1.551, 1.513, 1.648, 1.007, 0.927, 0.625],
        [0.535, 0, 1.434, 1.263, 1.566, 1.361, 0.742, 0.619],
        [1.551, 1.434, 0, 0.912, 0.512, 1.181, 1.145, 1.313],
        [1.513, 1.263, 0.912, 0, 0.568, 0.778, 1.474, 1.538],
        [1.648, 1.566, 0.512, 0.568, 0, 0.954, 1.227, 1.523],
        [1.007, 1.361, 1.181, 0.778, 0.954, 0, 1.547, 1.271],
        [0.927, 0.742, 1.145, 1.474, 1.227, 1.547, 0, 0.588],
        [0.625, 0.619, 1.313, 1.538, 1.523, 1.271, 0.588, 0]
    ]

change_sensor_name = \
{'7a00064336':          "",     # useless
 'Aircon0Power':        "",     # periodical-state
 'Aircon1Power':        "",     # ps
 'Color':               "",     # color
 'InferredPresence':     "",    # True/False, X
 'LightGroup1':          ["L01"], # ON/OFF
 'LightGroup2':          ["L02"], #
 'LightGroup3':          ["L03"], #
 'Luminosity':           "",        # X
 'Mtest1':               ["M01"],   #
 'Mtest2':              ["M02"],    #
 'Mtest3':              ["M03"],    #
 'Mtest4':              ["M04"],    #
 'Mtest5':              ["M05"],    #
 'Mtest6':              ["M06"],    #
 'Mtest8':              ["M07"],    #
 'Mtest9':              ["M08"],    #
 'Present':             "",         #
 'ProjectorInput':      "",         #
 'ProjectorPower':      ["P01"],      #
 'Screen':              "",         #
 'SeminarNumber':       "",         #
 'SetColor':            "",         #
 'SoundC':              "",         #
 'SoundCenter0':        ["SCN0"],
 'SoundCenter1':        ["SCN1"],
 'SoundCenter2':        ["SCN2"],
 'SoundWall0':          ["SWA0"],
 'SoundWall1':          ["SWA1"],
 'SoundWall2':          ["SWA2"],
 'SoundWindow0':        ["SWN0"],
 'SoundWindow1':        ["SWN1"],
 'SoundWindow2':        ["SWN2"],
 'StartAircon0':        ["AC0", "On"],
 'StartAircon1':        ["AC1", "On"],
 'StopAircon0':         ["AC0", "Off"],
 'StopAircon1':         ["AC1", "Off"],
 'TurnOff':      "",
 'TurnOffAirCon0':      ["AC0", "Off"],
 'TurnOffAirCon1':      ["AC1", "Off"],
 'TurnOffLightGroup1':      ["L01", "Off"],
 'TurnOffLightGroup2':      ["L02", "Off"],
 'TurnOffLightGroup3':      ["L03", "Off"],
 'TurnOffProjector':      ["P01", "Off"],
 'TurnOn':      "",
 'TurnOnLightGroup1':      ["L01", "On"],
 'TurnOnLightGroup2':      ["L02", "On"],
 'TurnOnLightGroup3':      ["L03", "On"],
 'TurnOnOff':      "",
 'TurnOnProjector':      ["P01", "On"],
 'seat1AOccupied':      ["S01"],
 'seat1BOccupied':      ["S02"],
 'seat2AOccupied':      ["S03"],
 'seat2BOccupied':      ["S04"],
 'seat3AOccupied':      ["S05"],
 'seat3BOccupied':      ["S06"],
 'seat4AOccupied':      ["S07"],
 'seat4BOccupied':      ["S08"],
 'seat5AOccupied':      ["S09"],
 'seat5BOccupied':      ["S10"],
 'seat6AOccupied':      ["S11"],
 'seat6BOccupied':      ["S12"],
 'seat7BOccupied':      ["S13"],
 'sensor0_Brightness':      "",
 'sensor0_Humidity':      "",
 'sensor0_Temperature':      "",
 'sensor1_Brightness':      "",
 'sensor1_Humidity':      "",
 'sensor1_Temperature':      "",
 'tiltAircon0_L_Pitch':      "",
 'tiltAircon0_L_Roll':      "",
 'tiltAircon0_Pitch':      "",
 'tiltAircon0_Roll':      "",
 'tiltAircon1_L_Pitch':      "",
 'tiltAircon1_L_Roll':      "",
 'tiltAircon1_Pitch':      "",
 'tiltAircon1_Roll':      "",
 'tiltTest_Pitch':      "",
 'tiltTest_Roll':      "",
 'totalSeatCount':      ""
}