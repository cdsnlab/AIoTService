import numpy as np
import os
import time
from .densityRatio import DensityRatio as dr

def change_point_detection(features, epsfolder, data_name='testbed', metric='SEP', save=False):
    n=2
    scores, thetas, sigmas, lambdas=[], [], [], []
    start=time.time()

    if save==True:
        metricfolder="{}/{}".format(epsfolder, metric)
        if not os.path.exists(metricfolder):
            os.mkdir(metricfolder)

    for t in range(len(features)):
        if t>0 and t%1000==0:
            print("{}/{} (accumulated) time: {}".format(t, len(features), time.time()-start))
            if save:
                np.save("{}/scores.npy".format(metricfolder), scores)
                np.save("{}/thetas.npy".format(metricfolder), thetas)
                np.save("{}/sigmas.npy".format(metricfolder), sigmas)
                np.save("{}/lambdas.npy".format(metricfolder), lambdas)
                # print("SAVE DONE.")

        prev1=max(0, t-1)
        post1=min(t+1, len(features)-1)
        post2=min(t+2, len(features)-1)
        
        numsensors=int((len(features[t])-12)/2)
        before=np.concatenate((features[t], features[post1])).reshape((n, -1))
        after=np.concatenate((features[post1], features[post2])).reshape((n, -1))

        before=before[:,12+numsensors:]
        after=after[:,12+numsensors:]

        if metric.lower()=="kliep":
            dre=dr(test_data=before, train_data=after); scores.append(dre.KLDiv)
        elif metric.lower()=='rulsif':
            dre=dr(test_data=before, train_data=after, alpha=0.1); scores.append(dre.PEDiv)
        elif metric.lower()=='ulsif':
            dre=dr(test_data=before, train_data=after); scores.append(dre.PEDiv)
        else:
            dre=dr(test_data=before, train_data=after); scores.append(dre.SEP)#; print(dre.SEP)
        if save:
            thetas.append(dre.theta)
            sigmas.append((dre.sigma, dre._median_distance))
            lambdas.append(dre.lambda_)

    # return scores, parameters
    if save:
        np.save("{}/scores.npy".format(metricfolder), scores)
        np.save("{}/thetas.npy".format(metricfolder), thetas)
        np.save("{}/sigmas.npy".format(metricfolder), sigmas)
        np.save("{}/lambdas.npy".format(metricfolder), lambdas)
        # print("TOTAL SAVE DONE.")

    return scores