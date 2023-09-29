# -*- coding: utf-8 -*-
from sklearn.neighbors import KDTree
from config import parameters
    
def get_solution_wrap(solution):    
    idx = [i for i, x in enumerate(solution) if x == 1]
    return idx

def get_sub_list(l, idx):
    result= [l[i] for i in idx]
    return result


     
def get_prediction_pool(pool, X_roc):
    y_pred = []
    for clf_base in pool:
        y_temp = clf_base.predict(X_roc)
        y_pred.append(y_temp)
    return y_temp


def get_roc_pred(pool, X, val_index, idx_roc):
    y_pool_roc = []
    for clf_base in pool:
        view_id = 0
        for bagging in clf_base:
            
            if (parameters.n_baggs > 1):
                bagg_predictors = bagging.estimators_
            else:
                bagg_predictors = [bagging]
            for bagg in bagg_predictors:
                X_dsel_view = X[view_id][val_index]
                X_roc = X_dsel_view[idx_roc]
                X_roc = [X_roc]
                y_bagg_temp = bagg.predict(X_roc)
                y_pool_roc.append(y_bagg_temp[0])
            view_id = view_id + 1
    
    return y_pool_roc


def get_roc_index(X, val_index, xq_index):
    roc_index = []
    roc_index = get_roc_index_view(X, val_index, xq_index)
    return roc_index

def get_roc_index_raw(X, val_index, xq_index):
    X_raw = X[0]
    X_dsel = X_raw[val_index]
    tree = KDTree(X_dsel, leaf_size=2) 
    X_xq = X_raw[xq_index]
    dist, idx_roc = tree.query([X_xq], k = parameters.k_roc)
    return idx_roc[0]

# Esse metodo pega para um Xq_index quem são os vizinhos mais próximos para cada view
# a partir de um KDTREE
# Definir o valor de k
def get_roc_index_view(X, val_index, xq_index):
    idx_roc_view = []
    for X_view in X:
        X_dsel = X_view[val_index]
        tree = KDTree(X_dsel, leaf_size=2) 
        # esse é o X_test
        X_xq = X_view[xq_index]
        # calculei a distância
        dist, idx_roc = tree.query([X_xq], k = parameters.k_roc)
        idx_roc_view.append(idx_roc[0])
    return idx_roc_view


# Retorna as predições dos classificadores selecionados
def get_pred_selecionados(y_pred_roc_x, idx_selecionados):
    y_pred_selecionados = []
    for i in idx_selecionados:
        pred_temp = y_pred_roc_x[i]
        y_pred_selecionados.append(pred_temp)
    return y_pred_selecionados
