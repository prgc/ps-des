from metrics import metrics_module
from predictions import predictions
import numpy as np
# In[]    
def get_metrics(preds, y_true,metrics):
    """
        This function reads the predictions made by the ensemble (any rule)
        and returns the metrics (auc, acc, f1)
                
        Parameters
        ----------
        preds : list of list int
            predictions list for each arrangement
        y_true : list of list int
            real classes ()
        metrics : string
            selected metric (auc, acc, f1)
        Returns
        -------
        metrics_list : list of list float
            list with the metrics
            
    """
    metrics_list = []
    #  para cada uma das x runs
    i = 0
    for runs in preds:
        metrics_list.append(metrics_module.calculate_metrics(runs, y_true[i], metrics))
        i = i + 1
    return metrics_list

# In[]    
def get_statistics(metrics_list,tipo):
    mean = 0
    if (tipo == 'mean') : 
            mean = np.average(metrics_list, axis=0)
    return mean