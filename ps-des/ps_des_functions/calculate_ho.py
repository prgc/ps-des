# calcula o HO
# require, todas as predições, as predições selecionados, e o y_true

def calculate_HO(prediction_all, prediction_sel, y_true):
    ho = []
    qtd_run = len(prediction_all)
    for run in range(qtd_run):
        ho_run = calculate_HO(prediction_all[run], prediction_sel[run], y_true[run])
        ho.append(ho_run) 
    return ho

def calculate_HO_run(prediction_all, prediction_sel, y_true_ds):
    i_xq = 0
    precision_ponderado = 0
    sum_beta = 0
    for prediction_xq in prediction_sel:
        y_true_xq = y_true_ds[i_xq]
        # maskedarray, transforma em uma array limpo
        prediction_xq_valid = list(prediction_xq.compressed())
        # calcula precision eq.1
        precision_temp = calculate_precision(prediction_xq_valid, y_true_xq)    
        # calcula beta eq.4
        beta_temp = calculate_beta(prediction_all[i_xq], y_true_xq)
        
        precision_ponderado = precision_ponderado + (precision_temp * beta_temp) 
        sum_beta = sum_beta + beta_temp
        i_xq = i_xq + 1
        
    ro = precision_ponderado / sum_beta
    return ro
# calcula prediction EQ. 1
def calculate_precision(prediction_xq, y_true):    
    # quantos classificadores selecionados acertaram o padrão
    qtd_competentes = prediction_xq.count(y_true)
    # total de classificadores selecionados
    qtd_selecionados = len(prediction_xq)
    precision = qtd_competentes/qtd_selecionados
    
    
    return precision
  
# calcula beta EQ. 4
def calculate_beta(all_predictions, y_true):
    all_predictions = list(all_predictions)
    qtd_acerto = all_predictions.count(y_true)
    tamanho_pool = len(all_predictions)
    beta = tamanho_pool - qtd_acerto
    return beta
  
# quantos classificadores competentes
def get_competent_clf(base_predictions, y_true):
    # conta quantos classificadores competentes estão no pool
    count_competent = base_predictions.count(y_true)
    return count_competent