import numpy as np
from ..info.config import config
def feature_label(cons_view, sensor_set, feature_name):
    """
        Input
            cons_view: numpy array (2*vs, F)
            sensor_set: set of total sensors in experiment
            feature_name: a dict of the names of features
        Output
            changed_label: the list of changed feature label
    """
    # changed_feature, 
    changed_label={}
    sensor_id={num:item for num, item in enumerate(sensor_set)}
    for col in range(cons_view.shape[1]):
        feature_col=cons_view[:,col]
        if max(feature_col)!=min(feature_col):
            # changed_feature.append(feature_col.reshape((-1,1)))
            if col<11:
                changed_label[col]=feature_name[col]
            else:
                if col>=11 and col<11+len(sensor_set):    # count
                    changed_label[col]="{}-{}".format(feature_name[11], sensor_id[col-11])
                else:   # time
                    changed_label[col]="{}-{}".format(feature_name[12], sensor_id[col-11-len(sensor_set)])
    # # changed_feature=np.concatenate(changed_feature, axis=1)

    return changed_label
