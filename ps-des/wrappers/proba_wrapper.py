import numpy as np


def get_sublist(lista, index):
    sublist = []
    for idx in index:
        sublist.append(lista[idx])
    return sublist


def get_proba_ds():
    print()



# tenho que dividir o proba por clf


# desmembra o proba_list salvo dos experimentos
def get_proba_wrapper(proba_list, clf_index):
    proba_wrapper_ds = []
    tam_run = len(proba_list)
    
    # para cada run
    for run_i in range(tam_run):
        proba_wrapper_run = []
        # a lista de probabilidaes para a run i
        proba_run = proba_list[run_i]
        # a lista de classificadores para a run i
        clf_run = clf_index[run_i]
        # quantidade de padrões de teste
        tam_xq = len(proba_run)
        # para todos os xq
        for xq in range(tam_xq):
            clf_idx_xq = clf_run[xq]
            # se o clf_index for = [], nenhum foi selecionado, logo pega o primeiro proba
            if (clf_idx_xq == []):
                proba_wrapper_run.append(proba_run[xq][0])
            # se classificadores foram escolhidos
            else:
                # pega apenas as predições reaizadas pelos classificadores escolhidos
                proba_ds = get_sublist(proba_run[xq], clf_idx_xq)
                proba_wrapper_run.append(proba_ds)
        proba_wrapper_ds.append(proba_wrapper_run)
        
    return proba_wrapper_ds


# retorna quais, para cada run, onde foram aplicados técnicas de seleção
def get_index_selection(selected_clf):
    idx_selected = []
    for run in selected_clf:
        idx_run = []
        idx = 0
        for sel_temp in run:
            if (sel_temp != []):
                idx_run.append(idx)
            idx = idx + 1
        idx_selected.append(idx_run)
    return idx_selected

# retorna apenas a predição feita pelos xq onde houve seleção
def proba_aggregration(proba_ds, selected_clf):
    proba_ds_new = []
    # os indices dos classificadores selecionados
    idx_ds = get_index_selection(selected_clf)
    qtd_run = len(proba_ds)
    # para cada run
    for run in range(qtd_run):
        proba_new_run = []
        proba_run = proba_ds[run]
        sel_run = idx_ds[run]
        # para cada xq
        for idx in sel_run:
            xq_proba = proba_run[idx]
            proba_new = np.array(xq_proba).transpose((1, 0, 2))
            proba_new_run.append(proba_new)
        proba_ds_new.append(proba_new_run)
    return proba_ds_new
        
        
        