import pickle
from modules import metrics_module, predictions

# In[]        
def load_predictions(base, arranjo,exp):
    """
        This function reads the results read in the @save_experiments and returns
        the predictions and the y_true. The results are stored using pickle api
                
        Parameters
        ----------
        base : string
        file's name
        Returns
        -------
        preds : list of list int
            predictions list for each arrangement
        y_true: list of list int
            real classes
            
    """
    string_exp = 'D://exp//exp' + str(exp) + '//'
    string_file = string_exp + arranjo + '/'+ base + ".txt"    
    # string_file = 'exp/' + arranjo + '/'+ base + ".txt"
    preds = pickle.load( open( string_file, "rb" ) )
    y_true = preds.pop()
    return preds,y_true

        
