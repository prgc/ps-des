import numpy as np
from config import parameters
from sklearn import metrics
from scipy.stats.mstats import gmean

def get_bagging(pool):
    bagging = []
    for clf_bagg in pool:
        bagg = clf_bagg[0].estimators_
        for bagg_temp in bagg:
            bagging.append(bagg_temp)
    return bagging

# return best des
def get_best_des(metric_list):
    max_value = max(metric_list)
    max_index = metric_list.index(max_value)
    return max_index



def get_des_info(des, xq):
    y_pred, base_predictions, selected_classifiers, ind_disagreement = des.predict([xq])
    # caso nenhum classificador tenha sido escolhido
    if (selected_classifiers == []):
        base_predictions_ds = base_predictions
        prediction_clf_selected = base_predictions.tolist()
        y_ds = []
        for i in base_predictions:
            y_ds.append(i[0])
        y_ds = np.array(y_ds)
    else:
        # class da ROC na qual houve DES
        y_ds = y_pred[ind_disagreement]
        # as predições onde houve DES
        base_predictions_ds = base_predictions[ind_disagreement]
        # os indexes dos classificadores selecionados apenas onde houve DES
        prediction_clf_selected = []
        i = 0
        # pegar apenas as predições dos classificadores selecionados
        # os indexes dos classificadores selecionados apenas onde houve DES
        selected_index = get_index_selected(selected_classifiers)
        for idx in selected_index:
            # wrapper in list para calcular o HO na classe HO
            prediction_clf_selected.append(base_predictions_ds[i][idx].tolist())
            i = i + 1
    
    return base_predictions_ds,prediction_clf_selected, y_ds



def calculate_potential_metric(data, metric):
    # base_predictions_ds,prediction_clf_selected, y_ds
    acc_list = []
    for des_data in data:
        base_predictions_ds = des_data[0]
        base_predictions_ds = base_predictions_ds.tolist()
        prediction_clf_selected = des_data[1]
        # prediction_clf_selected = prediction_clf_selected.tolist()
        y_ds = des_data[2]
        # y_ds = y_ds.tolist()
        # caso não tenha havido seleçãoo:
        if (len(base_predictions_ds[0]) == len(prediction_clf_selected[0])):
            acc_temp = 0.0
        else:
            y_true = []
            for i in range(len(prediction_clf_selected[0])):
                y_true.append(y_ds[0])
            if (metric == 'acc'):
                acc_temp = metrics.accuracy_score(y_true, prediction_clf_selected[0])
            if (metric == 'mcc'):
                y_true_array = np.array(y_true)
                acc_temp = metrics.matthews_corrcoef(y_true_array, prediction_clf_selected[0])
            if (metric == 'f1-macro'):
                acc_temp = metrics.f1_score(y_true, prediction_clf_selected[0], average='macro') 
            if (metric == 'gmean'):
                y_true_array = np.array(y_true)
                pred_array = np.array(prediction_clf_selected[0])
                acc_temp = gmean(y_true_array, pred_array) 
        acc_list.append(acc_temp)
    return acc_list

# retorna quais são as predições nas quais ocorreram seleção
def get_index_selected(selected_classifiers):
    idx_selected = []
    for clf_ROC in selected_classifiers:
        idx_temp = np.where(clf_ROC == True)
        idx_selected.append(idx_temp[0].astype(int))
    return idx_selected
    
# retorna as informações necessárias para calcular o HO
# todas as predições dos DES, as predições apenas dos clf selecionados
# e as classes dos elementos na ROC
def des_predictions(X_roc, des_algoritms):
    data = []
    # para cada algoritmo
    for des in des_algoritms:
        # para cada x_roc in X_roc
        for x in X_roc:
            y_pred, base_predictions, selected_classifiers, ind_disagreement = des.predict([x])
            # caso a técnica não escolha nenhum classificadores, vamos tratar como se ela tivesse
            # escolhido todos
            if (selected_classifiers == []):
                base_predictions_ds = base_predictions
                prediction_clf_selected = base_predictions.tolist()
                y_ds = []
                for i in base_predictions:
                    y_ds.append(i[0])
                y_ds = np.array(y_ds)
            else:
                y_ds = y_pred[ind_disagreement]
                # as predições onde houve DES
                base_predictions_ds = base_predictions[ind_disagreement]
                # os indexes dos classificadores selecionados apenas onde houve DES
                selected_index = get_index_selected(selected_classifiers)
                prediction_clf_selected = []
                i = 0
                # pegar apenas as predições dos classificadores selecionados
            for idx in selected_index:
                # wrapper in list para calcular o HO na classe HO
                prediction_clf_selected.append(base_predictions_ds[i][idx].tolist())
                i = i + 1
            data.append([base_predictions_ds,prediction_clf_selected, y_ds])
    return data        


# realiza a predição para o DES escolhido
def predict(X, DES, test_index):
    for view in X:
        y_pred, base_predictions, selected_classifiers, ind_disagreement = DES.predict([view[test_index]])
        idx_selected = get_index_selected(selected_classifiers)
        # caso nenhum seja selecionado, selecione todos
        if (idx_selected == []):
            idx_selected = list(range(0, parameters.tam_pool))
            return y_pred, idx_selected
        idx_selected = idx_selected[0].tolist()
            
    return y_pred, idx_selected
    