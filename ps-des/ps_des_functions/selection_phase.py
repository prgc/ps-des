import sys
sys.path.insert(1, "..")

import numpy as np
from config import parameters
from ps_des_functions import des_support

# return best des
def get_best_des(metric_list):
    max_value = max(metric_list)
    max_index = metric_list.index(max_value)
    return max_index


# return selected indexes
def get_index_selected(selected_classifiers):
    idx_selected = []
    for clf_ROC in selected_classifiers:
        idx_temp = np.where(clf_ROC == True)
        idx_selected.append(idx_temp[0].astype(int))
    return idx_selected

# get information from each des_algorithms
# informations are:
def get_des_data(X, test_index, des_algorithms):
    data_des = []
    # for each des
    for des in des_algorithms:
        # des data
        y_pred, base_predictions, selected_classifiers, ind_disagreement = des.predict([X[test_index]])
        # if no classifiers are selected
        if (selected_classifiers == []):
            base_predictions_ds = base_predictions
            prediction_clf_selected = base_predictions.tolist()
            y_ds = []
            for i in base_predictions:
                y_ds.append(i[0])
            y_ds = np.array(y_ds)
        else:
            # ROC class
            y_ds = y_pred[ind_disagreement]
            # des selected predictions
            base_predictions_ds = base_predictions[ind_disagreement]
            prediction_clf_selected = []
            i = 0
            # classifiers index
            selected_index = get_index_selected(selected_classifiers)
            for idx in selected_index:
                prediction_clf_selected.append(base_predictions_ds[i][idx].tolist())
                i = i + 1
        data = [base_predictions_ds,prediction_clf_selected, y_ds]
        data_des.append(data)
    return data_des


def selection(X, y, test_index, val_index, idx_roc, pred_all, des_algoritms):
    best_des = []
    X_raw = X[0]

    data_des = get_des_data(X_raw, test_index, des_algoritms)    

    if (parameters.potential_str == 'acc'):
        metric_list = des_support.calculate_potential_metric(data_des,'acc')
    if (parameters.potential_str == 'mcc'):
        metric_list = des_support.calculate_potential_metric(data_des, 'mcc')
    if (parameters.potential_str == 'f1'):
        metric_list = des_support.calculate_potential_metric(data_des, 'f1-macro')
    best_des = get_best_des(metric_list)
            
    return [], best_des