import torch

from utils.metrics import *
import os

from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score, confusion_matrix#, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, label_ranking_loss, coverage_error
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc



def network_eval(model, Iplus, Iminus, M, loader, dataset, write_folder, device, hyp):
    
    model.eval()

    for i, (x,y) in enumerate(loader):
          
        x = x.to(device)
        y = y.to(device)

        model.eval()

        constrained_output = model(x.float(), Iplus, Iminus, M, device)
        predicted = constrained_output.data > 0.5

        #Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to('cpu')
        cpu_constrained_output = constrained_output.to('cpu')
        y = y.to('cpu')
        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim =0)
    
    different_from_0 = (y_test.sum(0)!=0).clone().detach()
    y_test = y_test[:,different_from_0]
    constr_test = constr_test[:,different_from_0]
    predicted_test = predicted_test[:,different_from_0]

    hamming = hamming_loss(y_test, predicted_test)
    multilabel_accuracy = jaccard_score(y_test, predicted_test, average='micro')
    ranking_loss = label_ranking_loss(y_test, constr_test.data)
    avg_precision = label_ranking_average_precision_score(y_test, constr_test.data)
    cov_error = (coverage_error(y_test, constr_test.data) - 1) / constr_test.shape[1]
    one_err = one_error(y_test, constr_test.data)
 
    print("starting writing....")

    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    f = open(write_folder+"/"+dataset+'.csv', 'a')

    f.write(str(hyp['split'])+','+str(hyp['seed'])+ ',' +str(hyp['best_epoch']) + ',' +str(hyp['hidden_dim']) + ',' + str(hamming) + ',' + str(multilabel_accuracy) + ',' 
            + str(ranking_loss) + ',' + str(cov_error) + ',' + str(avg_precision) + ',' + str(one_err) + ',' + 'END' +  ',' + str(hyp) + '\n')

    f.close()