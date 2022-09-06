import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse


import torch
import torch.utils.data
import torch.nn as nn

import random

from utils.metrics import *
from utils.constraints_parser import *
from utils.val_handle import load_local
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, label_ranking_loss, coverage_error

from scipy.io import arff

from skmultilearn.dataset import available_data_sets
from skmultilearn.dataset import load_dataset

import numpy
import pickle

from model.network import *
from model.network_func import *

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc


parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--dataset', type=str, default='', metavar='S',
                    help='dataset to test (default: \'\')')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--split', type=int, default=0, metavar='Sp',
                    help='split of k-fold to use (default: 0)')
parser.add_argument('--epochs', type=int, default=100000,
                    help='numeber of epochs (default:100000)')
parser.add_argument('--device', type=str, default='0',
                    help='GPU (default:0)')
parser.add_argument('--num_classes', type=int, default='0',
                    help='Number of classes (default:0)')
parser.add_argument('--batch_size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--dropout', type=float, default=0.7, metavar='D',
                    help='dropout probability (default:0.7)')
parser.add_argument('--hidden_dim', type=int, default=3000, metavar='H',
                    help='size of the hidden layers (default: 3000)')
parser.add_argument('--num_layers', type=int, default=2, metavar='NH',
                    help='number of hidden layers (default: 2)')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='WD',
                    help='weight decay (default: 1e-5)')
parser.add_argument('--non_lin', type=str, default='relu', metavar='NL',
                    help='non linearity function to be used in the hidden layers (default: relu)')


args = parser.parse_args()
num_epochs = args.epochs

print("Running model", args)


# Load train, val and test set
dataset_name = args.dataset

# Set seed
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Set the hyperparameters
batch_size = args.batch_size
num_layers = args.num_layers
dropout = args.dropout
non_lin = args.non_lin
hidden_dim = args.hidden_dim
lr=args.lr
weight_decay =  args.weight_decay
max_patience = 20
path_identifier = 'seed_'+str(args.seed)+'_hidden_dim_'+str(args.hidden_dim)+'_weight_decay_'+str(args.weight_decay)+'_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_num_layers_'+str(args.num_layers)+'_dropout_'+str(args.dropout)+'_'+args.non_lin
path_identifier_no_seed = 'hidden_dim_'+str(args.hidden_dim)+'_weight_decay_'+str(args.weight_decay)+'_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_num_layers_'+str(args.num_layers)+'_dropout_'+str(args.dropout)+'_'+args.non_lin


# Set device
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
# Load data
file_name = './data/'+dataset_name+'/'+dataset_name+'_train'

if dataset_name == 'cal500' or dataset_name=='image' or 'rcv1subset' in dataset_name or dataset_name=='arts' or dataset_name=='business' or dataset_name=='science' or dataset_name=='computers' or dataset_name=='education' or dataset_name=='entertainment' or dataset_name=='health' or dataset_name=='social' or dataset_name=='society': 
    X, Y = load_local(dataset_name)
    X, testX, Y, testY = train_test_split(X, Y, test_size=0.30, random_state=0)
else:
    X, Y, feature_names, label_names = load_dataset(dataset_name, 'train')
    X, Y = X.todense(), Y.todense()
    testX, testY, feature_names, label_names = load_dataset(dataset_name, 'test')
    testX, testY = testX.todense(), testY.todense()
 
#Split it train and validation set 
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.15, random_state=seed)
# Preprocess the datasets
scaler = preprocessing.StandardScaler().fit((trainX.astype(float)))
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((trainX.astype(float)))
trainX = torch.tensor(scaler.transform(imp_mean.transform(trainX.astype(float)))).to(device).double()
trainY = torch.tensor(trainY).to(device).double()
valX = torch.tensor(scaler.transform(imp_mean.transform(valX.astype(float)))).to(device)
valY = torch.tensor(valY).to(device)
different_from_0 = (valY.sum(0)!=0).clone().detach()



#Create loaders 
train_dataset = [(x, y) for (x, y) in zip(trainX, trainY)]
val_dataset = [(x, y) for (x, y) in zip(valX, valY)]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



#Create model
model = ConstrainedFFNNModel(len(trainX[0]), hidden_dim, args.num_classes, num_layers,dropout,non_lin)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.BCELoss()

# Create matrices Iplus, Iminus and M
Iplus, Iminus = createIs('data/'+dataset_name+'/'+dataset_name+'_constraints.txt', args.num_classes)
Iplus, Iminus = torch.from_numpy(Iplus).to(device), torch.from_numpy(Iminus).to(device)
M = torch.from_numpy(createM('data/'+dataset_name+'/'+dataset_name+'_constraints.txt',args.num_classes)).to(device)


#Train the neural network
patience = max_patience
min_loss = 1e+300
loss_list = []
for epoch in range(num_epochs):
    model.train()

    for i, (x, labels) in enumerate(train_loader):

        model.float()

        x = x.to(device)
        labels = labels.to(device)
    
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        output = model(x.float(),Iplus, Iminus, M, device)

        train_output_plus = get_constr_out_train(output, labels, Iplus, Iminus, M, device, label_polarity='positive')
        train_output_minus = get_constr_out_train(output, labels, Iplus, Iminus, M, device, label_polarity='negative')
        train_output = (train_output_plus*labels)+(train_output_minus*(1-labels))    
        loss = criterion(train_output.double(), labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()

    
    if epoch%1==0:

        loss_list.append(loss)

        model.eval()

        val_loss = 0
        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            constrained_output = model(x.float(), Iplus, Iminus, M, device)
            val_loss += criterion(constrained_output.double(), y.double())
        val_loss /= float(i+1)

        if val_loss <= min_loss:
            patience = max_patience
            min_loss = val_loss
        else:
            patience = patience-1
  

        if patience==0:
            #Create folder to save hyperparameters if it does not exist
            if not os.path.exists('hyp/'+dataset_name):
                os.makedirs('hyp/'+dataset_name)
            if not os.path.exists('hyp/'+dataset_name):
                os.makedirs('hyp/'+dataset_name)
            #Save the hyperparameters + value of validation loss on file
            hyp = vars(args)
            hyp['best_epoch'] = epoch-max_patience
            hyp['val_loss'] = min_loss.item()
            dump_path = 'hyp/'+dataset_name+'/batch_size'+str(args.batch_size)+'_dropout'+str(args.dropout)+'_hdim'+str(args.hidden_dim)+'_lr'+str(args.lr)+'_nn_lin'+str(non_lin)+'_n_layers'+str(num_layers)+'_w_decay'+str(weight_decay)
            # Store data (serialize)
            with open(dump_path+'.pickle', 'wb') as handle:
                pickle.dump(hyp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            exit(0)

