import numpy as np
import os
import time
from .densityRatio import DensityRatio as dr
from .SEP import DensityRatioSEP as drSEP
from .info.config import config

def change_point_detection(features, metric='SEP'):
    n = config['vs']
    scores = []
    # start=time.time()

    for t in range(len(features)):
        # if t>0 and t%10000==0:
        #     print("{}/{} (accumulated) time: {}".format(t, len(features), time.time()-start))

        post1=min(t+1, len(features)-1)
        post2=min(t+2, len(features)-1)
        
        before=np.concatenate((features[t], features[post1])).reshape((n, -1))
        after=np.concatenate((features[post1], features[post2])).reshape((n, -1))

        if metric.lower()=="kliep":
            dre=dr(test_data=before, train_data=after); scores.append(dre.KLDiv)
        elif metric.lower()=='rulsif':
            dre=dr(test_data=before, train_data=after, alpha=0.1); scores.append(dre.PEDiv)
        elif metric.lower()=='ulsif':
            dre=dr(test_data=before, train_data=after); scores.append(dre.PEDiv)
        else:
            dre=drSEP(before, after); scores.append(dre.score)

    assert len(features)==len(scores)

    return scores