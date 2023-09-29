import sys
sys.path.append('../')
sys.path.insert(1, "C:/Users/roger/Dropbox/git/phd/trust_selection/")
from config import parameters
from save_load import load_experiments as load
from wrappers import proposal_wrapper as pw
from aggregation_module import aggregation_functions as agg
from wrappers import proba_wrapper
import numpy as np

def get_exp(conf):
    # caso 0-5 (50% test) (exp 41)
    # if ('mvl' in conf):
    #     exp = '_mvl'
    # elif ('svl' in conf or 'allclf' in conf or 'ho_des' in conf):
    #     exp = 'tese'
    # else:
    #     exp = 'tese'
    # return exp

    folder = 'C:\\exp\\exp'
    if ('mvl' in conf):
        exp = '_mvl'        
    # caso single view
    # exp = '46'
    exp = 'tese'
    if '0300' in conf:
        exp = exp + '\\' + 'homogeneo'
    if '0100' in conf:
        exp = exp + '\\' + 'het_weak'
    if '075' in conf:
        exp = exp + '\\' + 'het_strong'

    return exp

def get_data(conf,base):
    exp = get_exp(conf)        
    y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = load.load_data_dhes_ho(base,conf,exp)
    return y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view

    
    # if ('metades' in conf or 'desp' in conf or 'knora' in conf or 'knop' in conf):
    #     y_true, y_pred_temp, base_predictions_ds, prediction_selected, y_ds = load.load_data_ho(base,conf,exp)
    # elif('papper' in conf):
    #     y_true, y_pred_temp, selected_clf = load.load_data(base,conf,34)
    # elif('ho_des'):    
    #     y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = load.load_data_dhes_ho(base,conf,exp)
    #     return y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view
        



def get_diversity(conf,base, str_metric):
    
    exp = get_exp(conf)        
    if ('metades' in conf or 'desp' in conf or 'knora' in conf or 'knop' in conf):
        # y_true, y_pred_temp, base_predictions_ds, prediction_selected, y_ds = load.load_data_ho(base,conf,exp)
        y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = load.load_data_dhes_ho(base,conf,exp)
    elif('papper' in conf):
        y_true, y_pred_temp, selected_clf = load.load_data(base,conf,34)
    elif('ho' in conf or 'ga' in conf or 'mvl' in conf or 'selection' in conf):    
        y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = load.load_data_dhes_ho(base,conf,exp)
        # y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection = load.load_data_dhes_ho(base,conf,exp)
        
    # esses não precisam fazer o voto majoritario 
    if ('metades' in conf or 'desp' in conf or 'knora' in conf or 'knop' in conf):
        metric = pw.get_meta_des(y_true, y_pred_temp, str_metric)
    elif('papper' in conf):
        y_pred = agg.aggregation_predictions(y_pred_temp, [],'majority')
        metric = pw.get_meta_des(y_true, y_pred, str_metric)
    elif('ho' in conf or 'ga' in conf or 'mvl' in conf or 'selection' in conf):    
        proba_ds = proba_wrapper.get_proba_wrapper(proba, selected_clf)
        y_pred = agg.aggregation_predictions(y_pred_temp, proba_ds,'majority')
    return y_true, y_pred_all




def get_oracle_mvl(conf, base, str_metric):
    exp = get_exp(conf) 
    y_true, _, y_pred_all, _, _, _, _ = load.load_data_dhes_ho(base,conf,exp)
        
    pred_by_view = separate_pred_view(y_pred_all)
    
    views_oracle = []
    # separando por view
    i_view = len(parameters.views) - 1
    while (i_view >= 0):
    # for i_view in range(len(parameters.views)):
        # para cada run
        run_list = []
        for run in pred_by_view:
            xq_list = []
            for xq in run:
                xq_list.append(xq[i_view])
            run_list.append(xq_list)
        oracle = calculate_oracle(y_true, run_list, str_metric)
        views_oracle.append(oracle)
        
        i_view = i_view - 1
    # oracle_all = calculate_oracle(y_true, y_pred_all)
    # views_oracle.append(oracle_all)
    return views_oracle


def separate_pred_view(y_pred_all):
    qtd_clf = len(parameters.clf_base)
    qtd_views = len(parameters.views)    
    pred_views = []    
    # para cada run
    for y_run in y_pred_all:
        # para cada padrão
        pred_view_run = []
        for y_xq in y_run:
            # dividir por classificador
            clf_pred_list = np.array_split(y_xq, qtd_clf)
            # para cada classificador, temos 
            pred_view_temp = [ [] for _ in range(qtd_views) ]
            for y_clf in clf_pred_list:
                #divindo pela quantidade de bagging
                i_view = 0
                view_list = np.array_split(y_clf, qtd_views)
                for y_view in view_list:                    
                    # pred_view_temp[i_view].append(y_view)
                    if (pred_view_temp[i_view] == []):
                        pred_view_temp[i_view] = y_view
                    else:     
                        temp = pred_view_temp[i_view]
                        conc = np.concatenate((temp, y_view))
                        pred_view_temp[i_view] = conc              
                    i_view = i_view + 1
            # aqui eu tenho que rever as coisas
            pred_view_run.append(pred_view_temp)
        pred_views.append(pred_view_run)
    return pred_views


def calculate_oracle(y_true, y_pred_all, str_metric):
    qtd_run = len(y_true)
    
    y_oracle = []
    for run in range(qtd_run):
        y_oracle_run = []
        for i in range(len(y_true[run])):
            y_temp = y_pred_all[run][i]
            y_xq =  y_true[run][i]   
            # se existe um classificador selecionado que classifica
            # corretamente
            # caso no pool exista alguém que responder corretamente
            if (y_xq in y_temp):
                # escolha esse classificador
                y_oracle_temp = y_xq
            # caso contrario pegue qualquer elemento
            else:
                y_oracle_temp = y_temp[0]                        
            y_oracle_run.append(y_oracle_temp)
        y_oracle.append(y_oracle_run)
    metric = pw.get_meta_des(y_true, y_oracle, str_metric)
    return metric

def get_oracle_svl(conf,base, str_metric):
    exp = get_exp(conf) 
    y_true, _, y_pred_all, _, _, _, _ = load.load_data_dhes_ho(base,conf,exp)
    metric = calculate_oracle(y_true, y_pred_all, str_metric)
    return metric
    
def get_roc(conf,base):
    exp = get_exp(conf) 
    roc = load.load_roc(base,conf, exp)
    return roc

def get_metric(conf,base, str_metric):
    
    exp = get_exp(conf)        
    if('ho' in conf or 'ga' in conf or 'mvl' in conf or 'selection' in conf or 'ps' in conf):    
        y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = load.load_data_dhes_ho(base,conf,exp)
    # else ('metades' in conf or 'desp' in conf or 'knora' in conf or 'knop' in conf):
    else:
        y_true, y_pred_temp, base_predictions_ds, prediction_selected, y_ds = load.load_data_ho(base,conf,exp)
        
    if('ho' in conf or 'ga' in conf or 'mvl' in conf or 'selection' in conf or 'ps' in conf):    
        proba_ds = proba_wrapper.get_proba_wrapper(proba, selected_clf)
        y_pred = agg.aggregation_predictions(y_pred_temp, proba_ds,'majority')
        metric = pw.get_meta_des(y_true, y_pred, str_metric)
    # esses não precisam fazer o voto majoritario 
    # elif ('metades' in conf or 'desp' in conf or 'knora' in conf or 'knop' in conf):
    else:
        metric = pw.get_meta_des(y_true, y_pred_temp, str_metric)

    return metric


def return_clf_idx(clf_selected):
    
    qtd_bagg = parameters.n_baggs
    idx_list = []
    for clf in clf_selected:
        if (clf < qtd_bagg):
            idx_list.append(0)
        if (clf > qtd_bagg and clf < 100):
            idx_list.append(1)
        if (clf > 100):
            idx_list.append(2)
    return idx_list



def get_view_chooses(conf,base):
    # exp = get_exp(conf)
    y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = get_data(conf,base)
    
    chooses = []
    # para cada run
    for run in view:
        chooses_temp_run = []
        
        for views_choose in run:
            chooses_temp_run.append(views_choose)
        chooses.append(chooses_temp_run)
    return chooses



def get_clf_chooses(conf,base):
    # exp = get_exp(conf)
    y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, view = get_data(conf,base)
    
    chooses = []
    # para cada run
    for run in selected_clf:
        chooses_temp_run = []
        
        for clf_selected in run:
            idx_clf = return_clf_idx(clf_selected)
            chooses_temp_run.append(idx_clf)
        chooses.append(chooses_temp_run)
    return chooses



def get_des_chooses(conf,base):
    exp = get_exp(conf)
    y_true, y_pred_temp, y_pred_all, selected_clf, proba, des_selection, _ = get_data(conf,base)
    chooses = []
    for run in des_selection:
        chooses_temp_run = []
        for des in run:
            str_des = 'none'
            if ('METADES' in des):
                str_des = 'metades'
            if ('KNORAE' in des):
                str_des = 'knoraE'
            if ('KNORAU' in des):
                str_des = 'knoraU'
            if ('KNOP' in des):
                str_des = 'knop'
            if('DESP' in des):
                str_des = 'desp'                
            chooses_temp_run.append(str_des)
        chooses.append(chooses_temp_run)
    return chooses