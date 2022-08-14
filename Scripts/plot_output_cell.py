import os
from re import A
import torch
import torch.nn as nn
from cifar import CIFAR10
import matplotlib.ticker as ticker
import pandas as pd
from CKA import linear_CKA, kernel_CKA
import seaborn as sns
from scipy.stats import kendalltau
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from statics import create_model,load_dense_model,evaluate_model
# Loading the model
data = CIFAR10(False)
train_loader , test_loader = data.data_loaders()
#############################################
args={'exp_mode':'pretrain','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':1.0}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear

source_net_dense= "models/model_best_3for_dense_pretrain.pth.tar"
model = create_model(cl,ll,'darts_three_step',36,20,10)
model = load_dense_model(model,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model,'cpu',test_loader,criterion)
# print(top1,top5)
#########################################################
args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear
data = CIFAR10(False)
source_net_dense= "models/model_best_3for_dense_finetune.pth.tar"
model1 = create_model(cl,ll,'darts_three_step',36,20,10)
model1 = load_dense_model(model1,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model1,'cpu',test_loader,criterion)
# print(top1,top5)
###########################################################
model.eval()
model1.eval()
model.drop_path_prob = 0
model1.drop_path_prob = 0
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
for j in range(19):
    model.cells[j]._ops[7].register_forward_hook(get_activation('identity_12'))   
    model1.cells[j]._ops[7].register_forward_hook(get_activation('identity1_12')) 
    similarity_LCKA=0
    similarity_RCKA=0
    cos = nn.CosineSimilarity()
    model = model.to('cpu')
    model1 = model1.to('cpu')
    for data,label in test_loader:
        data=data.to('cpu')
        label = label.to('cpu')
        for i  in range(len(data)):
            model(torch.unsqueeze(data[i],0))
            output1 = activation['identity_12'].detach().numpy()
            avg_output1 = np.mean(output1, axis=(1,2))
            # output1 = torch.flatten(output1).detach().cpu().numpy()
            activation={}
            model1(torch.unsqueeze(data[i],0))
            output2 = activation['identity1_12'].detach().numpy()
            avg_output2 = np.mean(output2, axis=(1,2))
            # output2 = torch.flatten(output2).detach().cpu().numpy()
            activation={}
            # CKA
            a = linear_CKA(avg_output1, avg_output2)
            b = kernel_CKA(avg_output1, avg_output2)
            
            

            # tau, p_value = kendalltau(output1, output2)
            similarity_LCKA+=a 
            similarity_RCKA+=b
    print('Linear CKA: {}'.format(similarity_LCKA/len(test_loader)))    
    print('RBF Kernel CKA: {}'.format(similarity_RCKA/len(test_loader)))
############################################################




##############################################################
                #DARTS ORIG 120E 
###############################################################                
args={'exp_mode':'pretrain','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':1.0}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear

source_net_dense= "models/model_best_120E_dense_pretrain.pth.tar"
model = create_model(cl,ll,'darts_orig',36,20,10)
model = load_dense_model(model,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model,'cpu',test_loader,criterion)
# print(top1,top5)




# #########################################################
args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear
data = CIFAR10(False)
source_net_dense= "models/model_best_120E_desne_finetune.pth.tar"
model1 = create_model(cl,ll,'darts_orig',36,20,10)
model1 = load_dense_model(model1,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model1,'cpu',test_loader,criterion)
# print(top1,top5)
###################################################################
model.drop_path_prob = 0
model1.drop_path_prob = 0
activation = {}
model.eval()
model1.eval()
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
for j in range(19):
    model.cells[j]._ops[7].register_forward_hook(get_activation('identity_12'))   
    model1.cells[j]._ops[7].register_forward_hook(get_activation('identity1_12')) 
    similarity_LCKA=0
    similarity_RCKA=0
    cos = nn.CosineSimilarity()
    model = model.to('cpu')
    model1 = model1.to('cpu')
    for data,label in test_loader:
        data=data.to('cpu')
        label = label.to('cpu')
        for i  in range(len(data)):
            model(torch.unsqueeze(data[i],0))
            output1 = activation['identity_12'].detach().numpy()
            avg_output1 = np.mean(output1, axis=(1,2))
            # output1 = torch.flatten(output1).detach().cpu().numpy()
            activation={}
            model1(torch.unsqueeze(data[i],0))
            output2 = activation['identity1_12'].detach().numpy()
            avg_output2 = np.mean(output2, axis=(1,2))
            # output2 = torch.flatten(output2).detach().cpu().numpy()
            activation={}
            # CKA
            a = linear_CKA(avg_output1, avg_output2)
            b = kernel_CKA(avg_output1, avg_output2)
            
            

            # tau, p_value = kendalltau(output1, output2)
            similarity_LCKA+=a 
            similarity_RCKA+=b
    print('Linear CKA: {}'.format(similarity_LCKA/len(test_loader)))    
    print('RBF Kernel CKA: {}'.format(similarity_RCKA/len(test_loader)))
############################################################