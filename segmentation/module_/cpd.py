import time
import pickle as pkl
import pandas as pd
import numpy as np
from .win2feat import win2feat as wtf
from .densityratio import Densratio as dr
from .info.config import config, feature_name
from .helper.labeling import feature_label

def change_point_detection(windows, sensor_list, data_name='testbed', message=None, f_only=False):
    features, scores, params, changes=[], [], [], {}
    prev_mfs, max_wd=0,0
    ws, vs=config['ws'], config['vs']
    length=2 if data_name=='testbed' else 1

    sensor_type=set([item[:length] for item in sensor_list])#; print(sensor_type)

    start=time.time()
    for i in range(windows.shape[0]):
        t_n2, t_n1=max(0, i-2), max(0, i-1)
        t_p1, t_p2=min(windows.shape[0]-1, i+1), min(windows.shape[0]-1, i+2)

        # samples=np.concatenate(([windows[t_n2], windows[t_n1], windows[t_p1], windows[t_p2]]), axis=0).reshape((vs*2, ws, -1))
        # samples=np.concatenate(([windows[t_n1], windows[i], windows[t_p1]]), axis=0).reshape((vs+1, ws, -1))
        samples=np.concatenate(([windows[i], windows[t_p1], windows[t_p2]]), axis=0).reshape((vs+1, ws, -1))
        samples_, max_wd, prev_mfs=wtf(samples, sensor_list, max_wd, prev_mfs, sensor_type, data_name)
        samples_=np.array(samples_)
        features.append(samples_)#; changes[i]=changed_label
        #changed_label=feature_label(samples_, sensor_list, feature_name)

        if f_only:
            continue
        
        xt_n1, xt=samples_[:vs], samples_[vs-1:]
        
        dre=dr(x=xt_n1, y=xt)#; print(dre.SEPDiv, dre.sigma, dre.lambda_)
        

        

        scores.append(dre.SEPDiv); params.append((dre.sigma, dre.lambda_))

        if data_name=='hh101':
            if i!=0 and i%5000==0:
                print("{}/{}, elapsed time: {}".format(i, windows.shape[0]-1, time.time()-start))
                np.save("scores/{}_sep/f.npy".format(data_name), features)
                np.save("scores/{}_sep/s.npy".format(data_name), scores)
                np.save("scores/{}_sep/p.npy".format(data_name), params)
                # with open("scores/{}_sep/c.pkl".format(data_name), 'wb') as f:
                #     pkl.dump(changes, f)

        elif data_name=='adlmr':
            if message!=None:
                if i!=0 and i%5000==0:
                    print("{}/{}, elapsed time: {}".format(i, windows.shape[0]-1, time.time()-start))
                    np.save("scores/adlmr/f_{}.npy".format(message), features)
                    np.save("scores/adlmr/s_{}.npy".format(message), scores)
                    np.save("scores/adlmr/p_{}.npy".format(message), params)
            else:
                if i!=0 and i%5000==0:
                    print("{}/{}, elapsed time: {}".format(i, windows.shape[0]-1, time.time()-start))
                    np.save("scores/adlmr/f_raw.npy", features)
                    np.save("scores/adlmr/s_raw.npy", scores)
                    np.save("scores/adlmr/p_raw.npy", params)
        else:
            if i==int(windows.shape[0]/2):
                print('Half done', time.time()-start)
        

    return features, scores, params, changes