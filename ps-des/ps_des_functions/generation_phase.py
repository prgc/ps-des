import sys
sys.path.insert(1, "..")
from deslib.des import METADES, KNORAU, KNORAE, KNOP, DESP
from deslib.des.probabilistic import RRC
from config import parameters
from ps_des_functions import pool_train


def generation(X, y, train_index, val_index):
    pool = pool_train.train_ensemble(X,y,train_index)
    pool_bagging = pool_train.wrapper(pool)
    des_algoritms = initialize_des(X, y, val_index, pool_bagging)

    return pool, pool_bagging, des_algoritms



def initialize_des(X, y, val_index, pool):
    algorithms = parameters.des_algorithms
    X_raw = X[0]

    X_dsel = X_raw[val_index]
    y_dsel = y[val_index]

    # for each des, instatiante
    des_algs = []
    for alg in algorithms:
        des_temp = 0
        if (alg == 'rrc'):
            des_temp = RRC(pool)
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
