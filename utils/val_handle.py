import glob 
import pickle
import sys
from scipy.io import loadmat
from skmultilearn.dataset import load_dataset, load_from_arff


def get_best_hyp(dataset):
    """
    Returns the dictionary containing the best hyperparameters for each dataset 

    """
    min_val_loss = sys.float_info.max
    best_hyp = None
    load_folder = 'hyp/'+dataset+'/'
    load_files = glob.glob(load_folder+'*.pickle')
    for file in load_files:
        with open(file, 'rb') as handle:
            hyp = pickle.load(handle) 
            if hyp['val_loss'] < min_val_loss:
                min_val_loss = hyp['val_loss']
                best_hyp = hyp
    return best_hyp


def load_local(dataset_name):
    """"
    Load the datasets from local folder ./data

    """

    if dataset_name == 'cal500' or dataset_name == 'image' or dataset_name == 'arts' or dataset_name == 'business' or ('rcv1subset' in dataset_name) or dataset_name == 'science' or dataset_name == 'computers' or dataset_name =='education' or dataset_name == 'society' or dataset_name =='health' or dataset_name == 'social' or dataset_name =='entertainment':
        file_name = './data/'+dataset_name+'/'+dataset_name+'.mat'
        dic = loadmat(file_name)
        X, Y = dic['data'], dic['target']
        Y = Y.transpose()
        Y[Y < 0] = 0
        print(Y.shape)
    else:
        file_path = 'data/'+dataset_name+'/'+dataset_name+'.arff'
        label_count = {}
        label_count['cal500'], label_count['image'] = 174, 5
        nlab = {'arts':26,'business':30,'cal500':174,'emotions':6,'enron':53,'genbase':27,'image':5,'medical':45,'scene':6,'rcv1subset1':101,'rcv1subset2':101,'rcv1subset3':101,'rcv1subset4':101,'rcv1subset5':101,'science':40,'yeast':14}
        # Files are both dense
        arff_file_is_sparse = False
        # Files are in MULAN format -> labels appear at the end of the file
        label_location="end"
        X, Y = load_from_arff(file_path, label_count=label_count[dataset_name], label_location=label_location, load_sparse=arff_file_is_sparse)
        Y = Y.todense()
        print(Y.shape)

    return X, Y