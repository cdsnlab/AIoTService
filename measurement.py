import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# None(Skipped TW)
# earliness 계산
import pickle

milan_dir = ['221019-194009',
            '221019-200632',
            '221019-204541',
            '221019-215407',
            '221020-021007']
kyoto8_dir = ['221019-194114',
            '221019-195539',
            '221019-201452',
            '221019-203945',
            '221019-214809']
kyoto11_dir = ['221019-194117',
                '221019-202935',
                '221019-214613',
                '221020-002423',
                '221020-051013']
lapras_dir = ['221206-215426',
                '221206-215734',
                '221206-220250',
                '221206-221831',
                '221206-231118']

list_acc, list_earl1, list_earl2, list_HM1, list_HM2 = [], [], [], [], []
for dir in lapras_dir:
    # dir = '221019-194009'
    acc, earl1, earl2, HM1, HM2 = [], [], [], [], []
    # data.keys()
    for i in range(1, 4):
        with open(f'./output/log/{dir}/fold_{i}/dict_analysis.pickle', 'rb') as f:
            data = pickle.load(f)
        idx = np.where(data['lengths'] > 20)[0]
        accuracy = np.where(data['pred_y'][idx] == data['true_y'][idx], 1, 0).mean()
        earliness1 = np.mean((data['locations'][idx] - 20) / data['lengths'][idx])
        earliness2 = np.mean(data['locations'][idx] / data['lengths'][idx])
        acc.append(accuracy)
        earl1.append(earliness1)
        earl2.append(earliness2)
        HM1.append((2 * (1 - earliness1) * accuracy) / ((1 - earliness1) + accuracy))
        HM2.append((2 * (1 - earliness2) * accuracy) / ((1 - earliness2) + accuracy))
    list_acc.append(np.mean(acc))
    list_earl1.append(np.mean(earl1))
    list_earl2.append(np.mean(earl2))
    list_HM1.append(np.mean(HM1))
    list_HM2.append(np.mean(HM2))
print(list_acc)
print(list_earl1)
print(list_earl2)
print(list_HM1)
print(list_HM2)



# ---------------------------------------------------------------------------------------------
# Filter model
# earliness 계산
args = utils.create_parser()
args.dataset = "milan"
args.random_noise=True
data_natural = CASAS_RAW_NATURAL(args)


milan_dir = ['220926-154933']

locations = np.where(data_natural.lengths[data['idx']] <= 20, data_natural.lengths[data['idx']], data['locations']) 

list_acc, list_earl1, list_earl2, list_HM1, list_HM2 = [], [], [], [], []
for dir in milan_dir:
    # dir = '221019-194009'
    acc, earl1, earl2, HM1, HM2 = [], [], [], [], []
    # data.keys()
    for i in range(1, 4):
        data = pd.read_csv(f'./output/log/{dir}/fold_{i}/test_results.csv')
        
        
        locations = np.where(data['lengths'].to_numpy() <= 20, 1, data['locations'].to_numpy() - 20)        
        accuracy = np.where(data['pred_y'].to_numpy() == data['true_y'].to_numpy(), 1, 0).mean()
        earliness1 = np.mean(locations / data['lengths'].to_numpy())
        earliness2 = np.mean(data['locations'].to_numpy() / data['lengths'].to_numpy())
        acc.append(accuracy)
        earl1.append(earliness1)
        earl2.append(earliness2)
        HM1.append((2 * (1 - earliness1) * accuracy) / ((1 - earliness1) + accuracy))
        HM2.append((2 * (1 - earliness2) * accuracy) / ((1 - earliness2) + accuracy))
    list_acc.append(np.mean(acc))
    list_earl1.append(np.mean(earl1))
    list_earl2.append(np.mean(earl2))
    list_HM1.append(np.mean(HM1))
    list_HM2.append(np.mean(HM2))
print(list_acc)
print(list_earl1)
print(list_earl2)
print(list_HM1)
print(list_HM2)

0.936 / 0.948


lapras_dir = ['221206-180929',
            '221206-181539',
            '221206-182146',
            '221206-182912',
            '221206-190744']

kyoto11_dir = ['221004-234946']
list_acc, list_earl1, list_earl2, list_HM1, list_HM2 = [], [], [], [], []
for dir in lapras_dir:
    # dir = '221019-194009'
    acc, earl1, earl2, HM1, HM2 = [], [], [], [], []
    # data.keys()
    for i in range(1, 4):
        with open(f'./output/log/{dir}/fold_{i}/dict_analysis.pickle', 'rb') as f:
            data = pickle.load(f)
        
        locations = np.where(data['lengths'] <= 20, 1, data['locations'] - 20)        
        accuracy = np.where(data['pred_y'] == data['true_y'], 1, 0).mean()
        earliness1 = np.mean(locations / data['lengths'])
        earliness2 = np.mean(data['locations'] / data['lengths'])
        acc.append(accuracy)
        earl1.append(earliness1)
        earl2.append(earliness2)
        HM1.append((2 * (1 - earliness1) * accuracy) / ((1 - earliness1) + accuracy))
        HM2.append((2 * (1 - earliness2) * accuracy) / ((1 - earliness2) + accuracy))
    list_acc.append(np.mean(acc))
    list_earl1.append(np.mean(earl1))
    list_earl2.append(np.mean(earl2))
    list_HM1.append(np.mean(HM1))
    list_HM2.append(np.mean(HM2))
print(list_acc)
print(list_earl1)
print(list_earl2)
print(list_HM1)
print(list_HM2)
0.885/0.894

0.917/0.968


# acc = [0.8564, 0.8686, 0.8907]
# ear1 = [0.09182, 0.09609, 0.0929]
# ear2 = [0.1915, 0.1958, 0.1969]
acc = [0.7997, 0.781, 0.8102]
ear1 = [0.1146, 0.1005, 0.1042]
ear2 = [0.2121, 0.2123, 0.2072]

HM1, HM2 = [], []
for a, e1, e2 in zip(acc, ear1, ear2):
    HM1.append((2 * (1 - e1) * a) / ((1 - e1) + a))
    HM2.append((2 * (1 - e2) * a) / ((1 - e2) + a))
    
np.mean(HM1)
np.mean(HM2)
np.mean(ear2)

# ----------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format
dirs = ['221130-144706',
    '221130-145042',
    '221130-145456',
    '221130-150119',
    '221130-151746',
    '221206-172437',
    '221206-172801',
    '221206-173154',
    '221206-173649',
    '221206-174326',
    '221206-215426',
    '221206-215734',
    '221206-220250',
    '221206-221831',
    '221206-231118',
    '221211-210804',
    '221211-212709',
    '221211-221412',
    '221211-233902',
    '221212-025220']
dirs=['221206-180929',
'221206-181539',
'221206-182146',
'221206-182912',
'221206-190744']

dirs=['221211-220806',
'221211-221129',
'221211-221548',
'221211-222216',
'221211-223947',
'221207-181549',
'221207-181850',
'221207-182348',
'221207-183440',
'221207-190024',
'221207-145742',
'221207-150333',
'221207-150941',
'221207-151607',
'221207-154152']

dirs=['221213-132012',
'221213-132338',
'221213-132807',
'221213-133416',
'221213-134244',
'221213-135101',
'221213-135402',
'221213-135821',
'221213-140420',
'221213-141358',
'221213-142305',
'221213-142822',
'221213-143338',
'221213-143950',
'221213-150532']



for i, dir in enumerate(dirs):
    df = pd.read_csv(f'./output/log/{dir}/avg_results.csv')
    df = df[(df['step']==99)&(df['tag1']=='test')]
    a = df[(df['tag2']=='whole_accuracy')|(df['tag2']=='whole_earliness')|(df['tag2']=='whole_harmonic_mean')]['value'].to_list()
    print("%0.3f_%0.3f_%0.3f"%(a[0], a[1], a[2]))
    if (i+1) % 5 == 0:
        print(' ')
        


dirs=['221211-210804',
'221211-212709',
'221211-221412',
'221211-233902',
'221212-025220']

dirs=['221211-225216',
'221211-225954',
'221211-232950',
'221212-001620',
'221212-010310']

dirs=['221213-224556',
'221213-225225',
'221213-230941',
'221213-233630',
'221214-001759']


for i, dir in enumerate(dirs):
    df = pd.read_csv(f'./output/log/{dir}/avg_results.csv')
    df = df[(df['step']==99)&(df['tag1']=='test')]
    a = df[(df['tag2']=='whole_accuracy')|(df['tag2']=='whole_earliness_det')|(df['tag2']=='whole_harmonic_mean')]['value'].to_list()
    print("%0.3f_%0.3f_%0.3f"%(a[0], a[1], a[2]))
    if (i+1) % 5 == 0:
        print(' ')