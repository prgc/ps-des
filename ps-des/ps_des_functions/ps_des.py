import sys
sys.path.insert(1, "..")

from config import setup
from ps_des_functions import des_support, generation_phase, selection_phase, predictions, roc_functions



def start(base, run):
    y_pred = []
    y_true = []
    list_pred_all = []
    clf_sel = []
    y_proba_list = []
    des_list = []

    ############# Setup
    # define atributtes (X) and classes (Y)
    X, y, train_index, test_index, val_index = setup.setup_data(base,run)
    # generation phase
    pool, pool_bagging, des_algoritms = generation_phase.generation(X, y, train_index, val_index)
    
    # para cada xq    
    y_true = y[test_index]
    
    # dynamic selection, for each X_q
    for xq_index in test_index:
        # getting all predictions and prob for each clf for xq
        pred_all = predictions.get_predictions(pool, X, [xq_index])
        pred_all = [item for sublist in pred_all for item in sublist]
        proba_all = predictions.get_proba(pool, X, [xq_index])
        proba_all = [item for sublist in proba_all for item in sublist]
        
        # checking predictions
        if (predictions.check_same_predictions(pred_all)):
            # pega qualquer pred, já que todas são iguais
            y_xq = [pred_all[0]]
            # uma lista vazia para informar que não houve seleção
            idx_selected = []
            des_choosen = []
            idx_view = []
        # if predictions are not the same, selects
        else:        
            # RoC
            idx_roc = roc_functions.get_roc_index(X, val_index, xq_index) 

            # selection
            idx_view,idx_des = selection_phase.selection(X, y, xq_index, val_index, idx_roc, pred_all, des_algoritms)            
            # in case any of the DES
            if (idx_view == []):
                des_choosen = des_algoritms[idx_des]
                y_xq, idx_selected, = des_support.predict(X, des_choosen, xq_index)
            else:
                des_choosen = des_algoritms[idx_view][idx_des]
                y_xq, idx_selected, = des_support.predict([X[idx_view]], des_choosen, xq_index)
                
            # o y_xq tem que ser a saida escolhida pela técnica
            y_xq =  [pred_all[i] for i in idx_selected]

        # predict class 
        y_pred.append(y_xq)
        # all predictions
        list_pred_all.append(pred_all)
        # all prob
        y_proba_list.append(proba_all)        
        # selected classifier
        clf_sel.append(idx_selected)        
        # selected des
        des_list.append(str(des_choosen))
        
    return y_true, y_pred, list_pred_all, clf_sel, y_proba_list, des_list
