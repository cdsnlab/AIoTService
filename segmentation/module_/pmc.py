import numpy as np
import os
from matplotlib import pyplot as plt
from glob import glob

from .changePointDetection import change_point_detection
from .featureExtraction import sliding_window
from .dataLoader import dataLoader

def module_pmc(events, time_size, sensors):

    first_timestamp = float(events[0,2])
    sensortimestamps = {sensor: [first_timestamp] for sensor in sensors}
    tcorrelation_matrix = np.ones((len(sensors), len(sensors)))

    relationships = []

    for ei in range(len(events)): # Each Event in a Pair
        sensor, timestamp = events[ei, 0], float(events[ei, 2])

        sensortimestamps[sensor].append(timestamp)

        current_t = sensortimestamps[sensor]
        current_idx = sensors.index(sensor)

        for other in sensors:
            # if other==sensor: # Skip Itself
            #     continue
            other_t = sensortimestamps[other] if other!=sensor else sensortimestamps[other][:-1]

            # if len(current_t)>=time_size and len(other_t)>=time_size:
            if False:
                current_tmin = current_t[-time_size:]
                other_tmin = other_t[-time_size:]
            else:

                min_length = min(len(current_t), len(other_t))
                if min_length!=0:
                    current_tmin = current_t[-min_length:]
                    other_tmin = other_t[-min_length:]

                # current_omin = current_o[-min_length:]
                # other_omin = other_o[-min_length:]

            other_idx = sensors.index(other)

            if len(current_tmin)!=0:
                correlt = np.corrcoef(current_tmin, other_tmin)[0, 1]

            # correlo = np.corrcoef(current_omin, other_omin)[0, 1]
                if not np.isnan(correlt): # variance != 0
                    tcorrelation_matrix[current_idx, other_idx] = correlt
                    tcorrelation_matrix[other_idx, current_idx] = correlt
                
            # if not np.isnan(correlo): # variance != 0
            #     ocorrelation_matrix[current_idx, other_idx] = correlo
            #     ocorrelation_matrix[other_idx, current_idx] = correlo

        relationships.append(
            tcorrelation_matrix.copy()
            # np.mean(tcorrelation_matrix)
            # +np.mean(ocorrelation_matrix)
        )

    assert len(events)==len(relationships)

    return relationships


def correlation(dataset, metric):

    fig = plt.figure(figsize=(20, 5))
    #ax = plt.axes()

    window_size = 15
    time_size = 3

    if dataset=="hh101":
        episodes, transitions, labels = dataLoader(dataset)

        sensors = set()
        for episode in episodes:
            sensors = sensors | set(episode[:,0])
        sensors = sorted(sensors)

        for ti in range(len(episodes)):
            episode, transition, label = episodes[ti], transitions[ti], labels[ti]

            relationships = module_pmc(episode, time_size, sensors)

            mean_relationships = [np.mean(item) for item in relationships]

            np.save(f"./correlation/{dataset}/npy/{label}_correlation.npy", relationships)

            ax = fig.add_subplot(1, 1, 1)

            features = sliding_window(mean_relationships, window_size)

            values = []
            for i in range(len(features)):
                meanvalue = np.mean(features[i])
                diff = sum(abs(np.array(features[i])-meanvalue))
                values.append(diff)

            # np.save(f"./correlation/{dataset}/npy/{label}_diff.npy".format(dataset, label, ti), values)
            # np.save(f"./correlation/{dataset}/npy/{label}_label.npy".format(dataset, label, ti), [transition])

            plt.title(label)
            plt.bar(range(len(values)), values)
            plt.axvline(x=transition, linestyle=":", color="g", label="transition")
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.savefig(f"./correlation/{dataset}/png/{label}_diff.png",
                facecolor=fig.get_facecolor(), bbox_inches='tight' 
            )
            plt.clf()

            print("{}/{} {}".format(ti, len(episodes)-1, label))

            break

        return episode

        


    if dataset=="adlmr":

        episodes, transitions, labels = dataLoader(dataset)

        sensors = set()
        for episode in episodes:
            sensors = sensors | set(episode[:,0])
        sensors = sorted(sensors)

        for ti in range(len(episodes)):
            episode, transition, label = episodes[ti], transitions[ti], labels[ti]

            relationships = module_pmc(episode, time_size, sensors)

            mean_relationships = [np.mean(item) for item in relationships]

            np.save(f"./correlation/{dataset}/npy/{label}_correlation.npy", relationships)

            ax = fig.add_subplot(1, 1, 1)

            features = sliding_window(mean_relationships, window_size)

            values = []
            for i in range(len(features)):
                meanvalue = np.mean(features[i])
                diff = sum(abs(np.array(features[i])-meanvalue))
                values.append(diff)

            # np.save("./correlation/{}/npy/{}_diff.npy".format(dataset, label), values)
            # np.save("./correlation/{}/npy/{}_label.npy".format(dataset, label), [transition])

            plt.title(label)
            plt.bar(range(len(values)), values)
            for trans in transition:
                lb, ub = trans-1, trans
                tl, tr = float(episode[lb,2]), float(episode[ub,2])
                while tl-float(episode[lb,2])<5. and lb!=0:
                    lb-=1
                while float(episode[ub,2])-tr<5. and lb!=len(episode)-1:
                    ub+=1
                plt.axvline(
                    trans, linestyle="dotted", color="g"
                )
                plt.axvspan(
                    lb, ub, alpha=0.2, color='g'
                )
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.savefig("./correlation/{}/png/{}_diff.png".format(dataset, label),
                facecolor=fig.get_facecolor(), bbox_inches='tight'    
            )
            plt.clf()

            print("{}/{} {}".format(ti, len(episodes)-1, label))

            break

        return True

    
    if dataset=="check":
        total = 0
        transitions = sorted(glob("./correlation/*label.npy"))
        differences = sorted(glob("./correlation/*lstsq.npy"))

        assert len(transitions)==len(differences)

        for i in range(len(transitions)):
            diffs = np.load(differences[i])
            transition = np.load(transitions[i])[0]

            for j in range(50, transition-50):
                if diffs[j]>0.5:
                    total+=1

        return total


    if dataset=="testbed":

        episodes, transitions, labels = dataLoader(dataset)

        sensors = set()
        for episode in episodes:
            sensors = sensors | set(episode[:,0])
        sensors = sorted(sensors)

        result = 0
        for ti in range(len(episodes)): # Every Activity Pair
            episode, transition, label = episodes[ti], transitions[ti], labels[ti]
            folder = "./correlation/{}/{}".format(dataset, label)

            relationships = module_pmc(episode, time_size, sensors)

            mean_relationships = [np.mean(item) for item in relationships]

            np.save(f"./correlation/{dataset}/npy/{label}_correlation.npy", relationships)

            ax = fig.add_subplot(1, 1, 1)

            features = sliding_window(mean_relationships, window_size)

            values = []
            for i in range(len(features)):
                meanvalue = np.mean(features[i])
                diff = sum(abs(np.array(features[i])-meanvalue))
                values.append(diff)

            # np.save("./correlation/{}/npy/{}_diff.npy".format(dataset, label), values)
            # np.save("./correlation/{}/npy/{}_label.npy".format(dataset, label), [transition])

            plt.title(label)
            plt.bar(range(len(values)), values)
            for trans in transition:
                lb, ub = trans-1, trans
                tl, tr = float(episode[lb,2]), float(episode[ub,2])
                while tl-float(episode[lb,2])<5. and lb!=0:
                    lb-=1
                while float(episode[ub,2])-tr<5. and lb!=len(episode)-1:
                    ub+=1
                plt.axvline(
                    trans, linestyle="dotted", color="g"
                )
                plt.axvspan(
                    lb, ub, alpha=0.2, color='g'
                )
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            plt.savefig("./correlation/{}/png/{}_diff.png".format(dataset, label),
                facecolor=fig.get_facecolor(), bbox_inches='tight'    
            )
            plt.clf()

            print("{}/{} {}".format(ti, len(episodes)-1, label))

            break

            relationships = module_pmc(episode, time_size)

            assert len(episode)==len(relationships)

            ax = fig.add_subplot(1, 1, 1)

            features = sliding_window(relationships, window_size)

            values = []
            # lstsq = []
            for i in range(len(features)):
                meanvalue = np.mean(features[i])
                diff = sum(abs(np.array(features[i])-meanvalue))
                values.append(diff)

                # if len(values)>=window_size:
                #     y = np.array(values[-window_size:])
                #     A = np.vstack([np.array([k/10. for k in range(window_size)]), np.ones(window_size)]).T
                # else:
                #     y = np.array(values)
                #     A = np.vstack([np.array([k/10. for k in range(len(values))]), np.ones(len(values))]).T
                # m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                # lstsq.append(10*m if m>0 else 0)

            if not os.path.exists(folder): os.mkdir(folder)

            
            # np.save("./correlation/{}_lstsq.npy".format(label), lstsq)
            np.save("{}/episode.npy".format(folder), episode)
            np.save("{}/relationships.npy".format(folder), relationships)
            np.save("{}/diff.npy".format(folder), values)
            np.save("{}/label.npy".format(folder), [transition])

            plt.plot(range(len(relationships)), relationships)
            plt.axvline(x=transition, linestyle=':', color='g')
            plt.savefig("{}/relationships.png".format(folder),
                facecolor=fig.get_facecolor(), bbox_inches='tight'
            )
            plt.clf()

            plt.bar(range(len(values)), values)
            plt.axvline(x=transition, linestyle=':', color='g')
            plt.savefig("{}/diff.png".format(folder),
                facecolor=fig.get_facecolor(), bbox_inches='tight'
            )
            plt.clf()

            # plt.bar(range(len(lstsq)), lstsq)
            # plt.axvline(x=transition, linestyle=':', color='g')
            # plt.savefig("./correlation/{}_lstsq.png".format(label))
            # plt.clf()

            # flag=False
            # for j in range(transition-50, transition+50):
            #     if lstsq[j]>0.5:
            #         flag=True
            #         break
            # if flag:
            #     result+=1


            # scores = np.array(change_point_detection(features, metric))
            # if metric=="RuLSIF":
            #     scores[scores<0] = 0
            # assert len(scores) == len(relationships)
            # plt.bar(range(len(scores)), scores)
            # plt.axvline(x=transition, linestyle=':', color='g')
            # plt.savefig("./correlation/{}_{}.png".format(label, metric))
            # plt.clf()

            print("{}/{} {}".format(ti, len(episodes)-1, label))

    return result