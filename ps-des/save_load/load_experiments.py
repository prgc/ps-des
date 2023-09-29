import pickle
import sys
import os
sys.path.insert(1, "C:/Users/roger/Dropbox/git/phd/trust_selection/")

exp_folder = 'C://exp//exp'

# In[] pega todas as runs, e separa arranjo a arranjo
def wrapper_predictions(data):
    preds = data[0]
    # para cada run
    run_len = len(preds)
    for i in range(run_len):
        print()

# In[] pega todas as runs, e separa arranjo a arranjo
def wrapper_data(data, analise):
    
    tamanho_selection = len(data[0][0])
    
    preds = [ [] for _ in range(tamanho_selection) ]
    sel_clf = [ [] for _ in range(tamanho_selection) ]
    
    # para as predições
    for run in data[0]:
        for i in range(len(data[0][0])):
            preds[i].append(run[i])
    
    for run in data[1]:
        for i in range(len(data[1][0])):
            sel_clf[i].append(run[i])
    if (analise == 'trust'):
        trust = data[2]
        return preds, sel_clf,trust
 
    return preds, sel_clf


# In[] pega todas as runs, e separa arranjo a arranjo
# separa as configuracoes dado a qtd_clf
def split_preds(all_preds, all_clf, qtd_clf):
    # para cada run
    tam_runs = len(all_preds)
    
    clf_list = []
    preds_list =[]
    for runs in range(tam_runs):
        # para cada xq em cada run
        preds_xq_list = []
        clf_xq_list = []
        for xq in range(len(all_preds[runs])):    
            # a lista está em ordem reversa, os últimos são os maiores trust score e preds
            preds_reverse = all_preds[runs][xq][::-1]
            clf_reverse = all_clf[runs][xq][::-1]
            
            # separando as configuracoes dado a configuracao quantidade de classificadores de
            preds_reverse = preds_reverse[0:qtd_clf]
            preds_xq_list.append(preds_reverse)

            clf_reverse = clf_reverse[0:qtd_clf]
            clf_xq_list.append(clf_reverse)
        clf_list.append(clf_xq_list)
        preds_list.append(preds_xq_list)
    return preds_list, clf_list
    
# dado a configuracao com 20 classificadores, separa as predicoes e classificadores selecionados
#  para 1 ... n, configuracao de classificadores
def get_all_arran(preds,sel_clf):
    all_clf = sel_clf.pop()
    all_preds = preds.pop()
    
    all_arran_preds = []
    all_arran_clf = []
    tamanho_pool = len(all_clf[0][0])
    for qtd_clf in range(1,tamanho_pool+1):
        temp_pred, temp_clf = split_preds(all_preds, all_clf, qtd_clf)
        all_arran_preds.append(temp_pred)
        all_arran_clf.append(temp_clf)

    return all_arran_preds, all_arran_clf

def load_mvl(base, arranjo,exp):
    string_exp = exp_folder + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"
    # data[0] = predições por configuração
    # data[1] = y_true    
    data = pickle.load( open( string_file, "rb" ) )
   
    predictions = data[0]    
    proba = data[1]
    y_true = data[2]
    
    return predictions, proba, y_true


def load_data_ho(base, arranjo, exp):
    string_exp = exp_folder + str(exp) + '/'
    # data[0] = y_true
    # data[1] = y_pred    
    # data[2] = base_predictions_ds
    # data[3] = prediction_selected
    # data[4] = y_ds
    
    y_true = []
    y_pred = []
    base_predictions_ds = []
    prediction_selected = []
    y_ds = []
    string_file = string_exp + arranjo + '/'+ base +'/'
    size = len(os.listdir(string_file))
    if (size != 30):
        print(arranjo + ' ' + base)

    # for run in range(0,10):
    for run in range(0,30):
        string_file = string_exp + arranjo + '/'+ base +'/' + base + str(run) + ".txt"
        data = pickle.load( open( string_file, "rb" ) )    

        y_true_temp = data[0]
        y_pred_temp = data[1]    
        base_predictions_ds_temp =  data[2]
        prediction_selected_temp = data[3]
        y_ds_temp = data[4]

        
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp)
        base_predictions_ds.append(base_predictions_ds_temp)
        prediction_selected.append(prediction_selected_temp)        
        y_ds.append(y_ds_temp)
        
    return y_true, y_pred, base_predictions_ds, prediction_selected, y_ds


def load_roc(base, arranjo, exp):
    string_exp = exp_folder + str(exp) + '/'
    string_file = string_exp + arranjo + '/'+ base +'/'
    size = len(os.listdir(string_file))
    if (size != 30):
        print(arranjo + ' ' + base)

    # for run in range(0,10):
        
    roc = []        
    for run in range(0,30):
        string_file = string_exp + arranjo + '/'+ base +'/' + base + str(run) + ".txt"
        data = pickle.load( open( string_file, "rb" ) )    

        roc_temp = data[0]
        
        roc.append(roc_temp)
    return roc


# y_true, y_pred, y_pred_all, clf_sel
def load_data_dhes_ho(base, arranjo, exp):
    string_exp = exp_folder + str(exp) + '//'
    
    # data[0] = predições por configuração
    # data[1] = y_true    
    # data[2] = classificadores selecionados
    
    y_true = []
    y_pred = []
    y_pred_all = []
    clf_sel = []
    proba = []
    des_selection= []
    view = []
    
    string_file = string_exp + arranjo + '/'+ base +'/'
    size = len(os.listdir(string_file))
    if (size != 30):
        print(arranjo + ' ' + base)

    # for run in range(0,10):
    for run in range(0,30):
        string_file = string_exp + arranjo + '/'+ base +'/' + base + str(run) + ".txt"
        data = pickle.load( open( string_file, "rb" ) )
       
        y_true_temp = data[0]
        y_pred_temp = data[1]
        y_pred_all_temp = data[2]
        clf_sel_temp = data[3]
        proba_temp = data[4]
        
        clf_analises = data[5]
        # clf_analises = 0

        view_temp = data[len(data) - 1]
        
        
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp)
        y_pred_all.append(y_pred_all_temp)
        clf_sel.append(clf_sel_temp)
        proba.append(proba_temp)
        des_selection.append(clf_analises)
        view.append(view_temp)
    return y_true, y_pred, y_pred_all, clf_sel, proba, des_selection, view
    



def load_data(base, arranjo, exp):
    string_exp = exp_folder + str(exp) + '/'
    # data[0] = predições por configuração
    # data[1] = y_true    
    # data[2] = classificadores selecionados
    y_true = []
    y_pred = []
    clf_sel = []
    string_file = string_exp + arranjo + '/'+ base +'/'
    size = len(os.listdir(string_file))
    if (size != 30):
        print(arranjo + ' ' + base)

    # for run in range(0,10):
    for run in range(0,30):
        string_file = string_exp + arranjo + '/'+ base +'/' + base + str(run) + ".txt"
        data = pickle.load( open( string_file, "rb" ) )
       
        y_true_temp = data[0]
        y_pred_temp = data[1]    
        clf_sel_temp = data[2]
        
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp)
        clf_sel.append(clf_sel_temp)
        
    return y_true, y_pred, clf_sel
    

def load_meta_des(base, arranjo,exp):
    string_exp = exp_folder + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"
    # data[0] = predições por configuração
    # data[1] = y_true    
    data = pickle.load( open( string_file, "rb" ) )
   
    y_true = data[0]
    y_pred = data[1]    
    
    return y_true, y_pred



# pega todas as predições, cada posição é uma predição
def load_preds(base, arranjo,exp):

    string_exp = exp_folder + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"

    # data[0] = predições por configuração
    # data[1] = y_true    
    data = pickle.load( open( string_file, "rb" ) )
    y_true = data.pop()
    preds = wrapper_predictions(data)
    return preds, y_true



# pega todas as predições, cada posição é uma predição
def load_full(base, arranjo,exp):

    string_exp = exp_folder + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"

    # data[0] = predições por configuração
    # data[1] = labels dos classificadores selecionados por 
    # data[2] = trust
    # data[3] = y_true    
    data = pickle.load( open( string_file, "rb" ) )
    y_true = data.pop()
    preds, sel_clf, trust = wrapper_data(data,'trust')
    preds_full, clf_full = get_all_arran(preds, sel_clf)
    return preds_full,clf_full,trust,y_true

# pega o ultimo
def load_trust(base, arranjo,exp):

    string_exp = exp_folder + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"

    # data[0] = predições por configuração
    # data[1] = labels dos classificadores selecionados por 
    # data[2] = trust
    # data[3] = y_true    
    data = pickle.load( open( string_file, "rb" ) )
    y_true = data.pop()
    preds, sel_clf, trust = wrapper_data(data,'trust')
    return preds,sel_clf,trust,y_true
