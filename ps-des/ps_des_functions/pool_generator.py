import sys
sys.path.insert(1, "C:/Users/roger/Dropbox/git/phd/ps-des - Copia")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree, base
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron, LogisticRegression
from config import parameters
def add_list(lista_original, lista_temp):
    for i in range(len(lista_original)):
        lista_original[i].append(lista_temp[i])
# In[ ]: gera lista de lista
def gerar_lista_lista(tamanho):
    lista = []
    for i in range(tamanho):
        lista.append([])
    return lista
# In[ ]:
def break_list(views):
    lista_classificadores = []
    for i in views:
        for j in i:
            lista_classificadores.append(j)
    return lista_classificadores

# In[ ]: retorna um pool contendo os baggings completos
def get_bagging_views_ensemble(X,y,train_index):
    bagging_clf = []
    base_clf = get_base_classifiers()
    # para cada classificador base
    for clf in base_clf:
        temp = []
        for view in X:
            bagging = BaggingClassifier(clf,n_estimators=parameters.n_baggs)
            bagging.fit(view[train_index], y[train_index])
            temp.append(bagging)
        bagging_clf.append(temp)                    

    return bagging_clf      

# In[ ]:
def get_base_classifiers():
    """
        This function returns all the base classifiers which are going to be trained
        
        Parameters
        ----------

        Returns
        -------
        classifiers : list os base classifiers
            
    """
    pool = []
    for clf in parameters.clf_base:
        model = []
        if (clf == 'nb'):
            model = GaussianNB()
        if (clf =='perceptron'):
            model = CalibratedClassifierCV(Perceptron(max_iter=110))
            # model = Perceptron(max_iter=110)
        if (clf == 'lr'):
            model = CalibratedClassifierCV(LogisticRegression())        
            # model = LinearRegression()
        pool.append(model)
    return pool