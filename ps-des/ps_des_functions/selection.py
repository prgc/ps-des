import sys
sys.path.append('../')

import numpy as np
from deslib.des import METADES, KNORAU, KNORAE, KNOP, DESP, DESClustering, DESKNN, DESMI
from deslib.des.probabilistic import RRC
from config import parameters
from analise_ho import ho
from sklearn import metrics
import random
from scipy.stats.mstats import gmean
# import time

def train_des_single_view(X, y, val_index, pool):
    algorithms = parameters.des_algorithms
    X_raw = X[0]

    X_dsel = X_raw[val_index]
    y_dsel = y[val_index]

    # para cada algoritmo des, treinar com o dsel
    des_algs = []
    for alg in algorithms:
        des_temp = 0
        if (alg == 'rrc'):
            des_temp = RRC(pool)
        if (alg == 'desknn'):
            des_temp = DESKNN(pool)
        if (alg == 'cluster'):
            des_temp = DESClustering(pool)        
        if (alg == 'metades'):
            des_temp = METADES(pool)
        if (alg == 'knoraU'):
            des_temp = KNORAU(pool)
        if (alg == 'knoraE'):
            des_temp = KNORAE(pool)
        if (alg == 'knop'):
            des_temp = KNOP(pool)
        if (alg == 'desp'):
            des_temp = DESP(pool)
        des_temp.fit(X_dsel, y_dsel)
        des_algs.append(des_temp)        
    return des_algs


def train_des_mvl(X, y, val_index, pool):
    des_algs = parameters.des_algorithms
    qtd_views = len(parameters.views)
    # para cada view, treinar o clf_base
    des_view_list = []
    # para cada view
    for view_idx in range(qtd_views):

        # o pool para treinar
        pool_train_des_view = []
        # para cada conjunto de classificador no pool
        for clf_base in pool:
            # para cada um dos estimators
            for clf_bagg in clf_base[view_idx]:
                pool_train_des_view.append(clf_bagg)
        X_dsel = X[view_idx][val_index]
        y_dsel = y[val_index]
        # dado que tenho os classificadores treinados naquela view
        # agora é treinar os des para cada view

        des_view = []
        for des in des_algs:            
            des_temp = 0
            if (des == 'des_knn'):
                des_temp = DESKNN(pool_train_des_view)
            if (des == 'mi'):
                des_temp = DESMI(pool_train_des_view)
            if (des == 'knoraU'):
                des_temp = KNORAU(pool_train_des_view)
            if (des == 'metades'):
                des_temp = METADES(pool_train_des_view)
            if (des == 'knoraU'):
                des_temp = KNORAU(pool_train_des_view)
            if (des == 'knoraE'):
                des_temp = KNORAE(pool_train_des_view)
            if (des == 'knop'):
                des_temp = KNOP(pool_train_des_view)
            if (des == 'desp'):
                des_temp = DESP(pool_train_des_view)
            des_temp.fit(X_dsel, y_dsel)
            des_view.append(des_temp)        
        des_view_list.append(des_view)
    # retorna uma lista, onde cada elemento
    # é um DES treinados com classificadores de cada view
    return des_view_list

def train(X, y, val_index, pool):
    qtd_views = len(parameters.views)
    # caso svl
    if (qtd_views == 1):
        des = train_des_single_view(X, y, val_index, pool)
        return des
    # caso mvl
    else:
        des = train_des_mvl(X, y, val_index, pool)
        return des
        
def get_bagging(pool):
    bagging = []
    for clf_bagg in pool:
        bagg = clf_bagg[0].estimators_
        for bagg_temp in bagg:
            bagging.append(bagg_temp)
    return bagging

# retorna qual o melhor des pelo theta
def get_best_des_view(metric_list):
    # a melhor metrica encontrada
    best_metric = 0
    # a melhor view
    best_view = 0
    # a melhor des
    best_des = 0
    for idx_view in range(len(metric_list)):
        view_metrics = metric_list[idx_view]
        max_in_view = max(view_metrics)
        max_index = view_metrics.index(max_in_view)
        if (max_in_view > best_metric):
            best_metric = max_in_view
            best_view = idx_view
            best_des = max_index
    return best_view, best_des
    
    
    max_value = max(metric_list)
    max_index = metric_list.index(max_value)
    return max_index

# retorna qual o melhor des pelo theta
def get_best_des(metric_list):
    max_value = max(metric_list)
    max_index = metric_list.index(max_value)
    return max_index

def return_frequency_des(X, y, test_index, val_index, idx_roc, pred_all, des_algoritms, str_optimizer):
        X_raw = X[0]
        # o X para dsel e o X_ROC
        data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
        metric_list = calculate_HO_des(data_des)
        
        contador_igual = 0
        contador_diferente = 0
        contador_erro = 0
        output_clf = []
        for des in data_des: 
            output_temp = des[2]
            output_clf.append(output_temp[0])
           
           #saber se todos os elementos são iguais
        if(output_clf[1:] == output_clf[:-1]):
            if(metric_list[1:] != metric_list[:-1] or list(metric_list) == [0] * len(metric_list) or list(metric_list) == [1] * len(metric_list)):
                contador_igual = contador_igual + 1
                
        else:
            contador_diferente = contador_diferente + 1
        
        return contador_igual, contador_diferente


def return_frequency(X, y, test_index, val_index, idx_roc, pred_all, des_algoritms, str_optimizer):
        X_raw = X[0]
        # o X para dsel e o X_ROC
        data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
        metric_list = calculate_HO_des(data_des)
        
        contador_igual = 0
        contador_diferente = 0
        contador_erro = 0
        output_clf = []
        for des in data_des: 
            output_temp = des[2]
            output_clf.append(output_temp[0])
           
           #saber se todos os elementos são iguais
        if(output_clf[1:] == output_clf[:-1]):
            if(metric_list[1:] != metric_list[:-1] or list(metric_list) == [0] * len(metric_list) or list(metric_list) == [1] * len(metric_list)):
                contador_igual = contador_igual + 1
                
        else:
            contador_diferente = contador_diferente + 1
        
        return contador_igual, contador_diferente

# seleção baseada na proposta do melhor DES para aquele problema
def selection(X, y, test_index, val_index, idx_roc, pred_all, des_algoritms, str_optimizer):
    # caso tenha bagging
    best_view = []
    best_des = []
    qtd_views = len(parameters.views)
    if (qtd_views == 1):
        # pegar o primeiro elemento que é sempre a base crua
        X_raw = X[0]
        # o X para dsel e o X_ROC
        X_dsel = X_raw[val_index]
        X_roc = X_dsel[idx_roc]
        y_roc = y[val_index]
        y_roc = y[idx_roc]
        # informações sobre as classes da ROC
        # agora é escolher o DES com melhor ROC



        if (str_optimizer == 'ho_oracle'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_oracle(data_des, y[test_index])        

        if (str_optimizer == 'ho_des'):
            # pegar todas as informações que cada DES ofereceu para calcular o HO
            # lembrando que as informações foram retiradas para um x_q
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_des(data_des)

        if (str_optimizer == 'ho_random'):
            # pegar todas as informações que cada DES ofereceu para calcular o HO
            # lembrando que as informações foram retiradas para um x_q
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_random(data_des)


        if (str_optimizer == 'frequencia_des'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            calculate_frequencia(data_des)


            # metric_acc = calculate_HO_acc_teste(data_des,'acc')
            # metric_mcc = calculate_HO_acc_teste(data_des, 'mcc')
            # metric_f  = calculate_HO_acc_teste(data_des, 'f1-macro')            
            # print(metric_list)
            # print(metric_acc)
            # print(metric_mcc)
            # print(metric_f)
            # print()
            
        if (str_optimizer == 'ho_acc_teste'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_acc_teste(data_des,'acc')
        if (str_optimizer == 'ho_mcc_teste'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_acc_teste(data_des, 'mcc')
        if (str_optimizer == 'ho_f1_teste'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_acc_teste(data_des, 'f1-macro')
        if (str_optimizer == 'ho_gmean'):
            data_des = get_ho_des_data(X_raw, test_index, des_algoritms)    
            metric_list = calculate_HO_acc_teste(data_des, 'gmean')

            
        if (str_optimizer == 'ho_teste'):
            data_des = parameters.des_algorithms  
            # metric_list = calculate_HO_teste(data_des)            
        if (str_optimizer == 'ho_des_new'):
            # data_des = get_ho_des_data_new(X_raw, test_index, des_algoritms)  
            metric_list = calculate_HO_des_new(des_algoritms, X_roc, y_roc)            
        best_des = get_best_des(metric_list)
    
    # o que eu estava pensando a respeito de escolher os melhroes classificadores
    # para cada view e depois juntar tudo
    elif (parameters.str_optimizer == 'ho_des_new_'):
        all_metrics = []
        for view_idx in range(qtd_views):
            X_dsel = X[view_idx][val_index]
            X_roc = X_dsel[idx_roc[view_idx]]
            y_roc = y[val_index]
            y_roc = y[idx_roc[view_idx]]
            if (str_optimizer == 'ho_des'):
            # pegar todas as informações que cada DES ofereceu para calcular o HO
                data_des = get_ho_des_data(X[view_idx], test_index, des_algoritms[view_idx])    
                metric_list = calculate_HO_des(data_des)
            if (str_optimizer == 'ho_des_new'):
                data_des = get_ho_des_data_new(X[view_idx], test_index, des_algoritms[view_idx])                    
                metric_list = calculate_HO_des_new(des_algoritms[view_idx], X_roc, y_roc)              
            all_metrics.append(metric_list)
        best_view, best_des = get_best_des_view(all_metrics)
        
    # no caso do MVL | abordagem tradicional 
    else:
        all_metrics = []
        for view_idx in range(qtd_views):
            X_dsel = X[view_idx][val_index]
            X_roc = X_dsel[idx_roc[view_idx]]
            y_roc = y[val_index]
            y_roc = y[idx_roc[view_idx]]

            if (str_optimizer == 'ho_des'):
            # pegar todas as informações que cada DES ofereceu para calcular o HO
                data_des = get_ho_des_data(X[view_idx], test_index, des_algoritms[view_idx])    
                metric_list = calculate_HO_des(data_des)
            if (str_optimizer == 'ho_oracle'):            
                data_des = get_ho_des_data(X[view_idx], test_index, des_algoritms[view_idx])    
                metric_list = calculate_HO_oracle(data_des)        
            if (str_optimizer == 'ho_des_new'):
                metric_list = calculate_HO_des_new(des_algoritms[view_idx], X_roc, y_roc)              
            all_metrics.append(metric_list)
        best_view, best_des = get_best_des_view(all_metrics)
    return best_view, best_des



def split_ho_information( y_pred, base_predictions, selected_classifiers, ind_disagreement):
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


# a modificacao, vou pegar os classificadores selecionados
def get_ho_des_data_new(X, test_index, des_algorithms):
    data_des = []
    for des in des_algorithms:
        y_pred, base_predictions, selected_classifiers, ind_disagreement = des.predict([X[test_index]])
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
        data = [base_predictions_ds,prediction_clf_selected, y_ds]
        data_des.append(data)
    return data_des

# get information from each des_algorithms
# informations are:
def get_ho_des_data(X, test_index, des_algorithms):
    data_des = []
    # para cada algoritmo DES
    for des in des_algorithms:
        # pega as informações que o os algoritmos deram
        # start_time = time.time()
        y_pred, base_predictions, selected_classifiers, ind_disagreement = des.predict([X[test_index]])
        # print("DES PREDICTIONS --- %s seconds ---" % (time.time() - start_time))
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
            # start_time = time.time()
            selected_index = get_index_selected(selected_classifiers)
            # print("Pegando os classificadores selecionados --- %s seconds ---" % (time.time() - start_time))
            for idx in selected_index:
                # wrapper in list para calcular o HO na classe HO
                prediction_clf_selected.append(base_predictions_ds[i][idx].tolist())
                i = i + 1
        data = [base_predictions_ds,prediction_clf_selected, y_ds]
        data_des.append(data)
    return data_des



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



# qual o melhor DES
def calculate_HO_des_new(des_algoritms, X_roc, y_roc):
    ho_des_list = []
    for des in des_algoritms:
        des_hoa = []
        sum_beta = []
        sum_precision_ponderado = []
        for xq in X_roc:
            base_predictions_ds, prediction_clf_selected, y_ds = get_des_info(des, xq)
            base_predictions_ds = base_predictions_ds.tolist()
            if (len(base_predictions_ds[0]) == len(prediction_clf_selected[0])):
                # hoa_temp = 0.0
                precision_ponderado = 0
                beta = 0
            else:    
                # hoa_temp = ho.calculate_HO_dhes_run(base_predictions_ds, prediction_clf_selected, y_ds)
                precision_ponderado, beta = ho.calculate_HO_dhes_run(base_predictions_ds, prediction_clf_selected, y_ds)
    
            sum_precision_ponderado.append(precision_ponderado)
            sum_beta.append(beta)            
        
        if np.sum(sum_beta) == 0:
            hoa_temp = 0
        else:    
            hoa_temp = np.sum(sum_precision_ponderado)/np.sum(sum_beta)
        ho_des_list.append(hoa_temp)
        # ho_des_list.append(np.mean(des_hoa))
        
    return ho_des_list



# gera um valor aleatório
def calculate_HO_random(data):
    # pega as informações para cada des
    # base_predictions_ds,prediction_clf_selected, y_ds
    acc_list = []
    for des_data in data:
        acc_temp = random.uniform(0, 1)
        acc_list.append(acc_temp)
    return acc_list


# qual o melhor DES
def calculate_HO_oracle(data, y_true):
    # pega as informações para cada des
    # base_predictions_ds,prediction_clf_selected, y_ds
    oracle_list = []
    for des_data in data:
        
        y_ds = des_data[2]
        
        if (y_ds == y_true):
            oracle_temp = 1.0
        else:
            oracle_temp = 0.0
        oracle_list.append(oracle_temp)
    return oracle_list




# ESSE AQUI é o do artigo!
# qual o melhor DES
# def calculate_HO_acc_teste(data, metric):
#     # pega as informações para cada des
#     # base_predictions_ds,prediction_clf_selected, y_ds
#     acc_list = []
#     for des_data in data:
#         base_predictions_ds = des_data[0]
#         base_predictions_ds = base_predictions_ds.tolist()
#         prediction_clf_selected = des_data[1]
#         # prediction_clf_selected = prediction_clf_selected.tolist()
#         y_ds = des_data[2]
#         # y_ds = y_ds.tolist()
#         # caso não tenha havido seleçãoo:
#         # if (len(base_predictions_ds[0]) == len(prediction_clf_selected[0])):
#         #     acc_temp = 0.0

#         y_true = []
#         for i in range(len(prediction_clf_selected[0])):
#             y_true.append(y_ds[0])
#         if (metric == 'acc'):
#             acc_temp = metrics.accuracy_score(y_true, prediction_clf_selected[0])
#         if (metric == 'mcc'):
#             acc_temp = metrics.matthews_corrcoef(y_true, prediction_clf_selected[0])
#         if (metric == 'f1-macro'):
#             acc_temp = metrics.f1_score(y_true, prediction_clf_selected[0], average='macro') 

#         acc_list.append(acc_temp)
#     return acc_list


def calculate_HO_acc_teste(data, metric):
    # pega as informações para cada des
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
                # pred_array = np.array(prediction_clf_selected[0])
                acc_temp = metrics.matthews_corrcoef(y_true_array, prediction_clf_selected[0])
            if (metric == 'f1-macro'):
                acc_temp = metrics.f1_score(y_true, prediction_clf_selected[0], average='macro') 
            if (metric == 'gmean'):
                y_true_array = np.array(y_true)
                pred_array = np.array(prediction_clf_selected[0])
                acc_temp = gmean(y_true_array, pred_array) 

        acc_list.append(acc_temp)
    return acc_list



# qual o melhor DES
def calculate_frequencia(data):
    # pega as informações para cada des
    # base_predictions_ds,prediction_clf_selected, y_ds
    clf_selected = []
    for des_data in data:
        prediction_clf_selected = des_data[1]
        # y_ds = y_ds.tolist()
        # caso não tenha havido seleçãoo:
        clf_selected.append(prediction_clf_selected[0])
    

# qual o melhor DES
def calculate_HO_des(data):
    # pega as informações para cada des
    # base_predictions_ds,prediction_clf_selected, y_ds
    theta_list = []
    for des_data in data:
        base_predictions_ds = des_data[0]
        base_predictions_ds = base_predictions_ds.tolist()
        prediction_clf_selected = des_data[1]
        # prediction_clf_selected = prediction_clf_selected.tolist()
        y_ds = des_data[2]
        # y_ds = y_ds.tolist()
        # caso não tenha havido seleçãoo:
        if (len(base_predictions_ds[0]) == len(prediction_clf_selected[0])):
            theta_temp = 0.0
        else:    
            theta_temp = ho.calculate_HO_dhes_original_run(base_predictions_ds, prediction_clf_selected, y_ds)
            # theta_temp = ho.calculate_HO_dhes_run(base_predictions_ds, prediction_clf_selected, y_ds)
            
        theta_list.append(theta_temp)
    return theta_list

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
                # predições dos classificadores selecionados, os outros, tanto faz
                prediction_ds = base_predictions[ind_disagreement]
                # só as predições dos classificadores selecionados
                prediction_selected = np.ma.MaskedArray(prediction_ds, ~selected_classifiers)
                # class da ROC na qual houve DES
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
    