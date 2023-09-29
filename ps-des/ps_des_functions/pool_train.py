# This module implements functions which return the trained ensemble
import sys
# sys.path.insert(1, "C:/Users/roger/Dropbox/git/phd/ps-des - Copia")
# sys.path.append('C:/Users/roger/Dropbox/git/phd/ps-des')
sys.path.insert(1, "..")

from ps_des_functions import pool_generator
from config import parameters

   
# In[ ]: get
def get_bagging(pool):
    bagging = []
    for clf_bagg in pool:
        bagg = clf_bagg[0].estimators_
        for bagg_temp in bagg:
            bagging.append(bagg_temp)
    return bagging
# In[ ]: get
def train_ensemble(X,y,train_index):    
    """
        This function returns the trained ensemble for the dataset
        
        Parameters
        ----------
        X : list of float
            Generated views
        X_raw: float
            Raw data (uses only for bagging)
        y: list of int
            classes
        train_index: list of int
            train index
        Returns
        -------
        pool : list of classifiers
            generated ensemble
    """
    # caso bagging
    if (parameters.n_baggs > 1):
        pool = pool_generator.get_bagging_views_ensemble(X, y, train_index)
    # caso não bagging
    else:
        pool = pool_generator.get_views_ensemble(X, y, train_index)
    return pool

# In[ ]: get
def get_bagging_estimators(pool):
    qtd_views = len(parameters.views)
    bagging = []
    for clf_base in pool:        
        # caso só tenha uma view
        if (qtd_views == 1):
            bagg = clf_base[0].estimators_
            for bagg_temp in bagg:
                bagging.append(bagg_temp)
        # caso multiview
        else:   
            estimators_list = []
            for clf_bagg_view in clf_base:
                clf_estimator = clf_bagg_view.estimators_
                estimators_list.append(clf_estimator)
            bagging.append(estimators_list)
    return bagging      
# In[ ]: get
def wrapper(pool):
    if parameters.n_baggs>1:
        return get_bagging_estimators(pool)
    # para uma view
    else:
        pool_return = []
        for i in pool:
            pool_return.append(i[0])
        return pool_return
            