import sys
sys.path.insert(1, "..")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import bases_module, parameters
import numpy as np

def setup_data(base,run):


    X,y = bases_module.get_base(base)
    train_index, test_index, val_index = get_train_test_val(X, y, run)

    #####################################################################
    
    # split train e validation
    train_val_idx = train_index + val_index    
    X_train_val = X[train_val_idx]

    # normalize o train and validation
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X_train_val)
    X[train_val_idx] = X_normalizado
    X_test = X[test_index]

    # normalize test
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X_test)
    X[test_index] = X_normalizado
            
    return [X], y, train_index, test_index, val_index

def validation_class(y, val_index):
    classes = set(y[val_index])
    y_index = list(y[val_index])    
    for c in classes:
        qtd_elementos = y_index.count(c)
        if (qtd_elementos <= 3):
            return True
    return False

def get_train_test_val(X_raw, y , run):
    indices = range(len(X_raw))
    
    # dividir entre teste e validação
    validation_flag_train = True
    validation_flag_val = True
    
    while (validation_flag_train and validation_flag_val):
        X_train, X_test, y_train, y_test,train_index, test_index = train_test_split(X_raw, y, indices,test_size = parameters.test_size, stratify=y, random_state= run)        
        indices_train = range(len(X_train))
        X_train, X_val, y_train, y_val, new_train_index, val_index = train_test_split(X_train, y_train, indices_train,test_size = 0.5,random_state= run)
        train_index = np.array(train_index)        
        # separando corretamento os indexes
        val_index = train_index[val_index]
        train_index = train_index[new_train_index]
        val_index = list(val_index)
        train_index = list(train_index)            
        validation_flag_train = validation_class(y, train_index)
        validation_flag_val = validation_class(y, val_index)
    
    train_index = train_index + val_index
    split_test = np.array_split(test_index, 2)
    val_index = list(split_test[0])
    test_index = list(split_test[1])
    return train_index, test_index, val_index
