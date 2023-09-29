import sys
sys.path.insert(1, "..")

from ps_des_functions import ps_des
from save_load import save_experiments


# all datasets
datasets = ['appendicitis','australian','balance',
                  'cmc','column_3C','diabetes','glass1',
                  'glass6', 'haberman','hayes', 'heart',
                  'led7digit','mammographic','musk','pima','sonar',
                  'vehicle','vehicle2','vowel','wdbc']
for dataset_ in datasets:        
    begin = 0
    end = 30
    for run in range(begin,end):
        y_true, y_pred, y_pred_all, clf_sel, y_proba_list, des_list = ps_des.start(dataset_, run)            
        results = [y_true, y_pred, y_pred_all, clf_sel, y_proba_list, des_list]
        save_experiments.save_results(results , 'ps-des', dataset_, run)
