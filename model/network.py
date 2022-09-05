import torch
import torch.utils.data
import torch.nn as nn

import numpy as np


# Get the vector containing the value of each body having more than one literal
def get_v(Iplus, Iminus, out):

    out = out.unsqueeze(1)
    H = out.expand(len(out),len(Iplus),len(Iplus[0]))
    Iplus, Iminus = Iplus.unsqueeze(0), Iminus.unsqueeze(0)
    Iplus = Iplus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
    Iminus = Iminus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
    v, _ = torch.min((((Iplus*H)+(Iminus*(1-H)))+(1-Iplus-Iminus))*(1-Iplus*Iminus)+torch.min((Iplus*H),Iminus*(1-H))*Iplus*Iminus,dim=2)
    
    return v



#Get the value of the constrained output
def get_constr_out(x, Iplus, Iminus, M, device):
    
    c_out = x.double()
    v = get_v(Iplus, Iminus, c_out)
    
    # Concatenate the output of the network with the "value" associated to each body having more than one literal
    # and then expand it to a tensor [batch_size, num_classes, num_rules] <-- num rules stnads for the number of rules having len(body) > 1
    V = torch.cat((c_out,v), dim=1)
    V = V.unsqueeze(1)
    V = V.expand(len(x),len(Iplus[0]), len(Iplus[0])+len(v[0]))
    
    # Concatenate the matrix encoding the hierarchy (i.e., one literal rules) with the matrix M
    # which encodes which body corresponds to which head
    M = M.unsqueeze(0).double()
    R = torch.eye(len(Iplus[0])).unsqueeze(0).double().to(device)
    R = torch.cat((R,M),dim=2)
    R_batch = R.expand(len(x),len(Iplus[0]), len(Iplus[0])+len(v[0]))

    #Compute the final output
    final_out, _ = torch.max(R_batch*V.double(), dim = 2)
    return final_out



# Get the vector containing the value of each implicant having more than one literal for the training phase
def get_v_train(Iplus, Iminus, out, y, label_polarity):
    # H has shape (batch_size, num_rules, num_classes) <-- num rules stnads for the number of rules having len(body) > 1
    out = out.unsqueeze(1)
    H = out.expand(len(out),len(Iplus),len(Iplus[0]))
    
    y = y.unsqueeze(1)
    Y = y.expand(len(y),len(Iplus),len(Iplus[0]))

    Iplus, Iminus = Iplus.unsqueeze(0), Iminus.unsqueeze(0)
    Iplus = Iplus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
    Iminus = Iminus.expand(len(out),len(Iplus[0]),len(Iplus[0][0]))
    
    if label_polarity=='positive':
        vplus, _ = torch.min(Iplus*H*Y+(1-Iplus),dim=2)
        vminus,_ = torch.min(Iminus*(1-H)*(1-Y)+(1-Iminus),dim=2)
        v = torch.min(vplus,vminus)
    else:
        vplus, _ = torch.min(Iplus*H*(1-Y)+(1-Iplus)+Iplus*Y, dim=2)
        vminus,_ = torch.min(Iminus*(1-H)*Y+(1-Iminus)+Iminus*(1-Y), dim=2)
        v = torch.min(vplus,vminus)
    
    return v



#Get the value of the constrained output
def get_constr_out_train(x, y, Iplus, Iminus, M, device, label_polarity):
    
    assert(label_polarity=='positive' or label_polarity=='negative')
    out = x.double()
    v = get_v_train(Iplus, Iminus, out,y, label_polarity)

    y = y.unsqueeze(1)
    Y = y.expand(len(y),len(Iplus),len(Iplus[0]))   
    
    # Concatenate the output of the network with the "value" of each implicant having more than one literal
    # and then expand it to a tensor [batch_size, num_classes, num_rules]
    V = torch.cat((out,v), dim=1)
    V = V.unsqueeze(1)
    V = V.expand(len(x),len(Iplus[0]), len(Iplus[0])+len(v[0]))
    
    # Concatenate the matrix encoding the hierarchy (i.e., one literal rules) with the matrix M
    # which encodes which body corresponds to which head
    M = M.unsqueeze(0) 
    R = torch.eye(len(Iplus[0])).unsqueeze(0).double().to(device)
    R = torch.cat((R,M),dim=2)
    R_batch = R.expand(len(x),len(Iplus[0]), len(Iplus[0])+len(v[0]))
    # Compute the final output
    final_out, _ = torch.max(R_batch*V.double(), dim = 2)

    return final_out



class ConstrainedFFNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,dropout, non_lin):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = num_layers
        fc = []
        bn = []
        for i in range(self.nb_layers):
           if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
                bn.append(nn.BatchNorm1d(hidden_dim))
           elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
           else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
                bn.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.ModuleList(fc)
        self.bn = nn.ModuleList(bn)
        
        self.drop = nn.Dropout(dropout)
          
        self.sigmoid = nn.Sigmoid()
        if non_lin == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x, Iplus, Iminus, M, device):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f((self.fc[i](x)))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, Iplus, Iminus, M, device)
    
        return constrained_out