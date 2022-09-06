import os
import importlib
os.environ["DATA_FOLDER"] = "./"
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# If run on pssr add this line
plt.switch_backend('agg')

def plot_loss(loss,seed):
    fig, ax = plt.subplots()
    loss_v = [l.item() for l in loss]
    ax.plot(range(0,len(loss_v)*500,500), loss_v, color='red', linestyle='dashed', alpha=0.7, label="FFNN - rnd seed: " + str(seed))
    ax.set_xlabel("number of epochs")
    ax.set_ylabel("loss function")
    fig.savefig("./loss_10000epochs.png")
    return fig, ax

import argparse

import torch
import torch.utils.data
import torch.nn as nn

import random

from utils.metrics import *
from utils.constraints_parser import *
from utils.val_handle import *
from sklearn.datasets import load_svmlight_file

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, label_ranking_loss, coverage_error

from skmultilearn.dataset import available_data_sets
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split


import numpy
from model.network import *
from model.network_func import *

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--dataset', type=str, default='', metavar='S',
                    help='dataset to test (default: \'\')')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--device', type=str, default='0',
                    help='GPU (default:0)')
args = parser.parse_args()
# Take the hyperparameters value that generate the smallest validation loss
hyp = get_best_hyp(args.dataset)
num_epochs = hyp['best_epoch']
hyp['seed'] = args.seed


#Set seed
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")


# Load train, val and test set
dataset_name = args.dataset

if dataset_name == 'cal500' or dataset_name=='image' or 'rcv1subset' in dataset_name or dataset_name=='arts' or dataset_name=='business' or dataset_name=='science' or dataset_name=='computers' or dataset_name=='education' or dataset_name=='entertainment' or dataset_name=='health' or dataset_name=='social' or dataset_name=='society': 
    X, Y = load_local(dataset_name)
    #totalX, totalY = X.todense(), Y.todense()
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.30, random_state=0)
else:
    trainX, trainY, feature_names, label_names = load_dataset(dataset_name, 'train')
    trainX, trainY = trainX.todense(), trainY.todense()
    testX, testY, feature_names, label_names = load_dataset(dataset_name, 'test')
    testX, testY = testX.todense(), testY.todense()



#Added now 
#Split it train and validation set 
#trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.15, random_state=seed)

scaler = preprocessing.StandardScaler().fit((trainX.astype(float)))
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((trainX.astype(float)))
trainX = torch.tensor(scaler.transform(imp_mean.transform(trainX.astype(float)))).to(device).double()
trainY = torch.tensor(trainY).to(device).double()
testX = torch.tensor(scaler.transform(imp_mean.transform(testX.astype(float)))).to(device).double()
testY = torch.tensor(testY).to(device).double()
different_from_0 = torch.tensor((testY.sum(0)!=0))

#Set the hyperparameters
batch_size = hyp['batch_size']
num_layers = hyp['num_layers']
dropout = hyp['dropout']
non_lin = hyp['non_lin']
hidden_dim = hyp['hidden_dim']
lr= hyp['lr']
weight_decay =  hyp['weight_decay']
num_classes = int(hyp['num_classes'])


#Create loaders 
train_dataset = [(x, y) for (x, y) in zip(trainX, trainY)]
test_dataset = [(x, y) for (x, y) in zip(testX, testY)]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


#Create the model
model = ConstrainedFFNNModel(len(trainX[0]), hidden_dim, num_classes, num_layers,dropout,non_lin)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
w = torch.zeros(batch_size, num_classes, device=device).double()
criterion = torch.nn.BCELoss()

#Create the matrices for the CM module
Iplus, Iminus = createIs('data/'+dataset_name+'/'+dataset_name+'_constraints.txt', num_classes)
Iplus, Iminus = torch.from_numpy(Iplus).to(device), torch.from_numpy(Iminus).to(device)
M = torch.from_numpy(createM('data/'+dataset_name+'/'+dataset_name+'_constraints.txt',num_classes)).to(device)

#Train the neural network
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
        constr_output = get_constr_out(output, Iplus, Iminus, M, device)

        train_output_plus = get_constr_out_train(output, labels, Iplus, Iminus, M, device, label_polarity='positive')
        train_output_minus = get_constr_out_train(output, labels, Iplus, Iminus, M, device, label_polarity='negative')
        train_output = (train_output_plus*labels)+(train_output_minus*(1-labels))    
        loss = criterion(train_output.double(), labels)
       

        predicted = constr_output.data > 0.5

        # Total number of labels
        total_train = labels.size(0) * labels.size(1)
        # Total correct predictions
        correct_train = (predicted == labels.byte()).sum()

        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()


    if epoch%1==0:
        sys.stdout.write('\rEpoch: {}'.format(epoch))
        sys.stdout.flush()
print('\n')

model.eval()
network_eval(model, Iplus, Iminus, M, test_loader, dataset_name, 'results/', device, hyp)



