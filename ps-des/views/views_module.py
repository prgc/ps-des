# This module implements which manages views (representation)
import sys
sys.path.append('../')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from config import parameters
from sklearn import random_projection
# In[ ]: método que retorna apenas alguns indices de views
def select_views(views_completas, views_number):
    views = []
    for i in views_number:
        views.append(views_completas[i])
    return views
# In[ ]: Views names. Always put in order    
def get_views_names():
    return ['TSNE','PCA','Kernel_Linear','Kernel_Pol','Kernel_RBF']

# In[ ]:
# gera as views baseado nas representações. Retorna uma lista com as views (X) gerados
def generate_views(X):
    
    """
        This function creates all the views that will be used in the
        
        Parameters
        ----------
        X : list of float
            dataset features
        Returns
        -------
        views : list of floats
            a list which contains all the generated views
    """
    #tsne
    
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    
    # X = scaler.fit_transform(X)
        
    views_labels = parameters.views
    views = []
    for label in views_labels:
        if (label == 'raw'):
            views.append(X)
        if (label == 'tsne'):
            # T-SNE
            X_tsne = TSNE(n_components=3).fit_transform(X)
            # X_tsne = scaler.fit_transform(X_tsne)
            views.append(X_tsne)
        if (label == 'pca'):
            # PCA
            # pca_ = PCA(n_components='mle', svd_solver='full')
            pca_ = PCA(n_components=0.95, svd_solver='full')
            # pca_ = PCA()
            
            X_pca = pca_.fit_transform(X)
            # X_pca = scaler.fit_transform(X_pca)
            views.append(X_pca)
        if (label == 'kernel_linear'):
                # Kernels
            X_kernel_linear = metrics.pairwise.pairwise_kernels(X, metric ='linear')
            # X_kernel_linear = scaler.fit_transform(X_kernel_linear)
            views.append(X_kernel_linear)
        if (label == 'kernel_pol'):
            X_kernel_pol = metrics.pairwise.pairwise_kernels(X, metric='polynomial')
            # X_kernel_pol = scaler.fit_transform(X_kernel_pol)
            views.append(X_kernel_pol)
        if (label == 'kernel_rbf'):
            X_kernel_rbf = metrics.pairwise.pairwise_kernels(X, metric='rbf')
            # X_kernel_rbf = scaler.fit_transform(X_kernel_rbf)
            views.append(X_kernel_rbf)
            
        if (label == 'random_projection'):
            minimun = johnson_lindenstrauss_min_dim(X, eps=0.1)
            transformer = random_projection.GaussianRandomProjection(eps=0.5)
            X_random_projection = transformer.fit_transform(X)
            views.append(X_random_projection)
            
        if (label == 'rp_sparse'):
            transformer = SparseRandomProjection()
            X_random_projection = transformer.fit_transform(X)
            views.append(X_random_projection)
            
    
    return views
# In[ ]:
# gerar os vetores de caracteristicas (X), dado uma lista de views (definadas pelo método gerar_view) 
# e os indices de treino e teste
def gerar_X_views(views, train_index, test_index):    
    X_views_train = []
    X_views_test = []
    for v in views:
        X_views_train.append(v[train_index])
        X_views_test.append(v[test_index])    
    return X_views_train, X_views_test