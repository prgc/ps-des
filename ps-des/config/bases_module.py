import sys
sys.path.append('../')

from views import views_module as vf
import numpy as np

def get_predictions_proposal(base,views_index):
    path = path_generator(base)
    X, y = leitura_base(path)
    y_lista, y_pred = vf.get_views_proposal(X,y,views_index)    
    return y_lista, y_pred
# In[]    
def get_base(base):
    """
        This function reads the base
        
        Parameters
        ----------
        base : string
            The name of the dataset
        
        Returns
        -------
        X : np.array
            features of the dataset
        y : np.array
            classes of datasets
    """
    path = path_generator(base)
    X, y = leitura_base(path)
    return X, y


# In[]    
# Mudar para ter acesso a
def path_generator(base):
    # dataset's path
    
    sufix = '.data'
    # path_temp = general_configuration.dataset_folder + base + sufix
    path_temp = "bases/" + base + sufix
    return path_temp
# In[ ]:
# Retorna os atributos e classes de uma base
# 
def leitura_base(path):
    data = np.loadtxt(path,delimiter=',')
    tamanho = len(data[0])
    X = data[:,0:(tamanho-1)]
    y = data[:,tamanho-1]
    # transforma o array de double para int
    y = y.astype(int)
    return X,y
