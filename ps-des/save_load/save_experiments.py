import sys, os
sys.path.insert(1, "..")
import pickle
from config import parameters

def save_results(results, conf, base, run):
    
    
    folder = "../ps-des/results"
    if (not(os.path.exists(folder))):
        os.mkdir(folder)
    folder_exp = folder + '/' + parameters.potential_str + '/'  + base + '/'
    if (not(os.path.exists(folder_exp))):
        os.mkdir(folder_exp)
    string_file = folder_exp + base + str(run) +'.txt'    
    pickle.dump(results, open(string_file,"wb" ))
    