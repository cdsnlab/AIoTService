hh101_location={
    'D001': (10,5),
    'D002': (20,1),
    'D003': (28,6),
    'LS001': (22,2),
    'LS002': (4,4),
    'LS003': (4,9),
    'LS004': (4,14),
    'LS005': (15,11),
    'LS006': (6,2),
    'LS007': (6,4),
    'LS008': (13,19),
    'LS009': (26,15),
    'LS010': (19,5),
    'LS011': (26,9),
    'LS012': (31,19),
    'LS013': (21,13),
    'LS014': (29,10),
    'LS015': (29,8),
    'LS016': (5,11),
    'M001': (22,2),
    'M002': (4,4),
    'M003': (4,9),
    'M004': (4,14),
    'M005': (15,11),
    'M006': (6,2),
    'M007': (6,4),
    'M008': (13,19),
    'M009': (26,15),
    'M010': (19,5),
    'M011': (26,9),
    'M012': (31,19),
    'MA013': (21,13),
    'MA014': (29,10),
    'MA015': (29,8),
    'MA016': (5,11),
    'T101': (10,5),
    'T102': (20,1),
    'T103': (28,6),
    'T104': (10,6),
    'T105': (10,6),
}

hh_act={
    "": "Other",
    "Bathe": "Bathe",
    "Bed_Toilet_Transition": "Bed_Toilet_Transition",
    "Cook": "Cook",
    "Cook_Breakfast":"Cook",
    "Cook_Dinner":"Cook",
    "Cook_Lunch":"Cook",
    "Dress":"Other",
    "Eat":"Eat",
    "Eat_Breakfast":"Eat",
    "Eat_Dinner":"Eat",
    "Eat_Lunch":"Eat",
    "Enter_Home":"Enter_Home",
    "Entertain_Guests":"Other",
    "Evening_Meds":"Other",
    "Groom":"Other",
    "Leave_Home":"Leave_Home",
    "Morning_Meds":"Other",
    "Personal_Hygiene":"Personal_Hygiene",
    "Phone":"Other",
    "Read":"Other",
    "Relax":"Relax",
    "Sleep":"Sleep",
    "Sleep_Out_Of_Bed":"Sleep",
    "Toilet":"Other",
    
    'Take_Medicine': 'Other',
    'Exercise': 'Other',
    'Drink': 'Other',
    'Step_Out': 'Other',
    'Housekeeping': 'Other',

    'Caregiver': 'Other', 
    'Drug_Management': 'Other', 
    'Groceries': 'Other', 
    'Laundry': 'Other', 
    'Make_Bed': 'Other', 
    'Paramedics': 'Other', 
    'Piano': 'Other', 
    'System_Technicians': 'Other', 

    "Wash_Breakfast_Dishes":"Wash_Dishes",
    "Wash_Dinner_Dishes":"Wash_Dishes",
    "Wash_Dishes":"Wash_Dishes",
    "Wash_Lunch_Dishes":"Wash_Dishes",
    "Watch_TV":"Other",
    'Work': 'Work',
    'Work_At_Desk': 'Work',
    "Work_At_Table":"Work",
    'Work_On_Computer': 'Work',
    'g1.Caregiver': 'Other',
    'g1.Housekeeping': 'Other', 
    'g1.Inspection': 'Other', 
    'g1.Maintenance': 'Other', 
    'g1.Movers': 'Other', 
    'r1.Cook_Breakfast': 'Cook', 
    'r1.Dishes': 'Wash_Dishes', 
    'r1.Dress': 'Other', 
    'r1.Groom': 'Other', 
    'r1.Personal_Hygiene': 'Personal_Hygiene', 
    'r1.Relax': 'Relax', 
    'r1.Toilet': 'Other', 
    'r1.Watch_TV': 'Other', 
    'r1.Work': 'Work'

 }

hh_sensors=['BATP001', 'BATP002', 'BATP003', 'BATP004', 'BATP005', 'BATP006', 'BATP007', 'BATP008', 'BATP009', 'BATP010', 'BATP011', 'BATP012', 'BATP013', 'BATP014', 'BATP015', 'BATP016', 'BATP017', 'BATP018', 'BATP019', 'BATP020', 'BATP021', 'BATP022', 'BATP023', 'BATP024', 'BATP025', 'BATP026', 'BATP027', 'BATP028', 'BATP101', 'BATP102', 'BATP103', 'BATP104', 'BATP105', 'BATP106', 'BATV001', 'BATV002', 'BATV003', 'BATV004', 'BATV005', 'BATV006', 'BATV007', 'BATV008', 'BATV009', 'BATV010', 'BATV011', 'BATV012', 'BATV013', 'BATV014', 'BATV015', 'BATV016', 'BATV017', 'BATV018', 'BATV019', 'BATV020', 'BATV021', 'BATV022', 'BATV023', 'BATV024', 'BATV025', 'BATV026', 'BATV027', 'BATV028', 'BATV101', 'BATV102', 'BATV103', 'BATV104', 'BATV105', 'BATV106', 'D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D101', 'F001', 'L001', 'L002', 'L003', 'L004', 'L005', 'L006', 'L007', 'L008', 'L009', 'L010', 'LL001', 'LL002', 'LL005', 'LL007', 'LL008', 'LS001', 'LS002', 'LS003', 'LS004', 'LS005', 'LS006', 'LS007', 'LS008', 'LS009', 'LS010', 'LS011', 'LS012', 'LS013', 'LS014', 'LS015', 'LS016', 'LS017', 'LS018', 'LS019', 'LS020', 'LS021', 'LS022', 'LS023', 'LS024', 'LS025', 'LS026', 'LS027', 'LS028', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'MA001', 'MA002', 'MA003', 'MA004', 'MA005', 'MA006', 'MA007', 'MA008', 'MA009', 'MA010', 'MA011', 'MA012', 'MA013', 'MA014', 'MA015', 'MA016', 'MA017', 'MA018', 'MA019', 'MA020', 'MA021', 'MA022', 'MA023', 'MA024', 'MA025', 'MA026', 'MA027', 'MA028', 'T101', 'T102', 'T103', 'T104', 'T105', 'T106', 'T107', 'T108', 'T109', 'T110', 'ZB001', 'ZB002', 'ZB003', 'ZB004', 'ZB005', 'ZB006', 'ZB007', 'ZB008', 'ZB009', 'ZB010', 'ZB011', 'ZB012', 'ZB013', 'ZB014', 'ZB015', 'ZB101', 'ZB102', 'ZB103', 'ZB104', 'ZB105', 'ZigbeeNetSecCounter']