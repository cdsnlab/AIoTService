import numpy as np
import os
import time
from .densityRatio import DensityRatio as dr

def change_point_detection(features, order, pairname, data_name='testbed', metric='SEP', save=False):
    n=2
    scores, thetas, sigmas, lambdas=[], [], [], []
    start=time.time()

    activity_folder="./outputs/{}/{}/{}".format(data_name, metric, pairname)
    episode_folder="{}/{}".format(activity_folder, order)

    if not os.path.exists(activity_folder):
        os.mkdir(activity_folder)
    if not os.path.exists(episode_folder):
        os.mkdir(episode_folder)

    for t in range(len(features)):
        if t>0 and t%1000==0:
            print("{}/{} (accumulated) time: {}".format(t, len(features), time.time()-start))
            if save:
                np.save("{}/scores.npy".format(episode_folder), scores)
                np.save("{}/thetas.npy".format(episode_folder), thetas)
                np.save("{}/sigmas.npy".format(episode_folder), sigmas)
                np.save("{}/lambdas.npy".format(episode_folder), lambdas)
                print("SAVE DONE.")

        prev1=max(0, t-1)
        post1=min(t+1, len(features)-1)
        post2=min(t+2, len(features)-1)

        before=np.concatenate((features[prev1], features[t])).reshape((n, -1))
        after=np.concatenate((features[post1], features[post2])).reshape((n, -1))
        # after=np.concatenate((features[t], features[post1])).reshape((n, -1))

        # if metric.lower()=="kliep":
        #     dre=dr(test_data=before, train_data=after, option=metric.lower()); scores.append(dre.KLDiv)
        # elif metric.lower()=='rulsif':
        #     dre=dr(test_data=before, train_data=after, option=metric.lower(), alpha=0.5); scores.append(dre.PEDiv)
        # else:
        dre=dr(test_data=before, train_data=after); scores.append(dre.SEP)#; print(dre.SEP)
        if save:
            thetas.append(dre.theta)
            sigmas.append((dre.sigma, dre._median_distance))
            lambdas.append(dre.lambda_)

    # return scores, parameters
    if save:
        np.save("{}/scores.npy".format(episode_folder), scores)
        np.save("{}/thetas.npy".format(episode_folder), thetas)
        np.save("{}/sigmas.npy".format(episode_folder), sigmas)
        np.save("{}/lambdas.npy".format(episode_folder), lambdas)
        print("TOTAL SAVE DONE.")

    return scores