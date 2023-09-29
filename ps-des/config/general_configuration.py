from sys import platform

if platform == 'win32':
    dataset_folder = 'C:/bases//'
    # experiment_save_folder = 'C://exp//exp40//'
    experiment_save_folder = 'C://exp//exp'
    slash = '//'
    # configuration_folder = 'C://conf//'

if platform == 'linux' or platform =='linux2':
    experiment_save_folder = '/home/proger/exp/exp'
    # experiment_save_folder = '/home/proger/exp/exp'
    slash = '/'
    dataset_folder = '/home/proger/bases/'
    # configuration_folder = '/home/proger/conf/'
