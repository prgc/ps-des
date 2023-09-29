import sys
sys.path.append('../')

from metrics import metrics_module
from predictions import predictions
from deslib.util import aggregation
from sklearn import metrics
import numpy as np
from scipy.stats.mstats import gmean
# In[]    
def get_oracle(preds,y_true):
    """
        This function reads the predictions made by the arragements and returns
        prediction made by majoritary vote. Using dynamic ensemble selection
                
        Parameters
        ----------
        preds : list of list int
            predictions list for each arrangement
        Returns
        -------
        moracle_list : list of list int
            predictions made by majoritary vote in Dynamic ensemble
            each position of the list contais one specific the resultos
            of a specific arrangement
            
    """
    oracle_list = []
    # para cada arranjo experimental (e.g. tamanho do ensemble)
    # arranjo tem 'n' runs
    for arranjo in preds:
        temp = []
        i = 0
        for ensemble_pred in arranjo:
            pred_temp = predictions.get_oracle_des(ensemble_pred, y_true[i])
            temp.append(pred_temp)
            i = i + 1
        oracle_list.append(temp)
    return oracle_list

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
    for runs in preds:
        i = 0
        means_temp = []
        for parametros in runs:
            means_temp.append(metrics_module.get_metrics(parametros, y_true[i], metrics))
            i = i + 1
        metrics_list.append(means_temp)
    # mean_list = calculate_mean(metrics_list)
    return metrics_list

def get_meta_des(y_true, y_pred, str_metric):
    qtd_run = len(y_true)
    metric_list = []
    metric_temp = 0
    for run in range(0,qtd_run):
        if (str_metric == 'acc'):
            metric_temp = metrics.accuracy_score(y_true[run], y_pred[run])
        if (str_metric == 'f1-macro'):
            metric_temp = metrics.f1_score(y_true[run], y_pred[run], average='macro') 
        if (str_metric == 'f1-micro'):
            metric_temp = metrics.f1_score(y_true[run], y_pred[run], average='micro')             
        if (str_metric == 'gmean'):
            metric_temp = gmean(y_true[run], y_pred[run]) 
        if (str_metric == 'mcc'):
            metric_temp = metrics.matthews_corrcoef(y_true[run], y_pred[run])
        metric_list.append(metric_temp)
    return metric_list


# In[]    
def get_majoritary_des(preds):
    """
        This function reads the predictions made by the arragements and returns
        prediction made by majoritary vote. Using dynamic ensemble selection
                
        Parameters
        ----------
        preds : list of list int
            predictions list for each arrangement
        Returns
        -------
        majoritary_list : list of list int
            predictions made by majoritary vote in Dynamic ensemble
            each position of the list contais one specific the resultos
            of a specific arrangement
            
    """
    majoritary_list = []
    # para cada arranjo experimental (e.g. tamanho do ensemble selecionado)
    for arranjo in preds:
        
        temp = []
        for ensemble_pred in arranjo:
            pred_temp = predictions.get_majoritary_predictions_des(ensemble_pred)
            temp.append(pred_temp)
            
        
        majoritary_list.append(temp)
    return majoritary_list

def get_(preds):
    # para cada arranjo experimental (e.g. tamanho do ensemble selecionado
    y_total = []
    for pred_runs in preds:
        # para cada run, pegar as predições de cada clf_base, e formar um novo
        size = len(pred_runs[0])
        preds_majo_temp = []
        for i in range (size):
            i_element  = [item[i] for item in pred_runs]
            preds_majo_temp.append(i_element)
    y_total.append(preds_majo_temp)
    return y_total
# In[]    
def get_majoritary(preds):
    """
        Parameters
        ----------
        preds : list of list int
            predictions list for each arrangement
        Returns
        -------
        majoritary_list : list of list int
            predictions made by majoritary vote in Dynamic ensemble
            each position of the list contais one specific the resultos
            of a specific arrangement
            
    """
    majoritary_list = []
    # para cada arranjo experimental (e.g. tamanho do ensemble selecionado)
    for pred_runs in preds:
        
        # para cada run, pegar as predições de cada clf_base, e formar um novo
        size = len(pred_runs[0])
        preds_majo_temp = []
        for i in range (size):
            i_element  = [item[i] for item in pred_runs]
            preds_majo_temp.append(i_element)
        
        pred_majo = predictions.get_majoritary_predictions_des(preds_majo_temp)
        majoritary_list.append(pred_majo)
    
    return majoritary_list

# In[]   
def get_aggregation_rule(y_proba, aggregation_rule):
    list_proba = []
    for run_proba in y_proba:
        # no formato para o DESLIB
        format_proba = np.array(run_proba).transpose((1, 0, 2))
        if (aggregation_rule == 'max'):
            proba_temp = aggregation.maximum_rule(format_proba)
        if (aggregation_rule == 'min'):
            proba_temp = aggregation.minimum_rule(format_proba)
        if (aggregation_rule == 'median'):
            proba_temp = aggregation.median_rule(format_proba)
        if (aggregation_rule == 'product'):
            proba_temp = aggregation.product_rule(format_proba)            
            
        list_proba.append(proba_temp)
    return list_proba
   
# In[] 
def get_statistics(metrics_list,tipo):
    """
        This function return an list with statistics values (e.g. mean, std)
        given an list with metrics (e.g. auc, acc, f1)         
        Parameters
        ----------
        metrics_list : list of list float
            metrics list
        tipo : string
            type of statistics to be calculated (e.g. mean, std)
        Returns
        -------
        mean_list: list of float
            a statitiscs list
            
    """
    mean_list = []
    for metrics in metrics_list:
        mean_list.append(metrics_module.calculate_statistics(metrics,tipo))
    return mean_list
