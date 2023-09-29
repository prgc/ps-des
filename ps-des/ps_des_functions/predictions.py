# This module implements functions which deals with the predictions
import sys
sys.path.insert(1, "..")
from config import parameters


def check_same_predictions(preds):
    a =  (preds.count(preds[0]) == len(preds))
    return a
# In[ ]:
def gerar_lista_lista(tamanho):
    lista = []
    for i in range(tamanho):
        lista.append([])
    return lista

# In[ ]:# predicoes do pool
def get_view_predictions(pool, X_view, test_index):
    prediction_list = []
    for bag in pool:
        pred_temp = bag.predict(X_view[test_index])
        prediction_list.append(pred_temp)
    return prediction_list
# In[ ]:
# proba do pool
def get_view_proba(pool, X_view, test_index):
    proba_list = []
    for bag in pool:
        proba_temp = bag.predict_proba(X_view[test_index])
        proba_list.append([proba_temp])
    return proba_list


# In[ ]: predictions
def get_predictions(pool, views, test_index):
    prediction_list = []
    # for each base classifier
    for clf_base in pool:
        # for each view
        qtd_views = len(views)
        for i in range(qtd_views):
            # bagging
            if (parameters.n_baggs > 1):
                # bagging estimators
                bagg_estimators = clf_base[i].estimators_
                preds = get_view_predictions(bagg_estimators, views[i], test_index)
            else:    
                preds = get_view_predictions([clf_base[i]], views[i], test_index)
            ###############################
            for pred_temp in preds:
                prediction_list.append(pred_temp)
    return prediction_list    
    # In[ ]: bagging
def get_proba(pool, views, test_index):
    proba_list = []
    # for each base classifier
    for clf_base in pool:
        # views
        qtd_views = len(views)
        for i in range(qtd_views):
            # case bagging
            if (parameters.n_baggs > 1):
                bagg_estimators = clf_base[i].estimators_
                proba = get_view_proba(bagg_estimators, views[i], test_index)
            else:
                proba = get_view_proba([clf_base[i]], views[i], test_index)
            for proba_temp in proba:
                proba_list.append(proba_temp)
    return proba_list    

