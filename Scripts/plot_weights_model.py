import os
import torch
import torch.nn as nn
from cifar import CIFAR10
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
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

source_net_dense= "models/model_best_pt_dense_darts_14_108.pth.tar"
model = create_model(cl,ll,'darts_three_step',108,14,10)
model = load_dense_model(model,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model,'cpu',test_loader,criterion)
# print(top1,top5)
#########################################################
args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear
data = CIFAR10(False)
source_net_dense= "models/model_best_ft_dense_darts_14_108.pth.tar"
model1 = create_model(cl,ll,'darts_three_step',108,14,10)
model1 = load_dense_model(model1,source_net_dense,'cpu',args)
# top1,top5 = evaluate_model(model1,'cpu',test_loader,criterion)
# print(top1,top5)
# ##################################################

model.drop_path_prob = 0
##############################################
weights = model.cells[13]._ops[2].op[1].weight
print(weights.shape)
w = weights.flatten().detach().numpy()
wp=pd.DataFrame(w,columns=['data_pt'])
########################################
weights1 = model1.cells[13]._ops[2].op[1].weight
print(weights1.shape)
w1 = weights1.flatten().detach().numpy()
w1_idx = np.where(w1!=0)
# w1_idx_non = np.where(w1!=0)
z=w1[w1_idx]

# w_c = np.column_stack((w,w1)).T
# print(w_c.shape)
# 
# wp1=pd.DataFrame(w1,columns=['data_ft'])
###################################################
# w_c_p= pd.DataFrame({'data_pt':w_c[0,:],'data_ft':w_c[1,:]}).fillna(0)
# print(w_c)
# sns.displot(w_c_p, x="data_pt", hue="data_ft", fill=True)
fig, ax = plt.subplots()
ax.hist(w, bins=200, edgecolor='black', alpha=0.3)
ax.hist(z, bins=10, edgecolor='black', alpha=0.3)
ax.set_title("Histogram")
ax.set_xlim(min(w.min(),w1.min()),max(w.max(),w1.max()))
ax.set_xlabel("X axis")
ax.set_ylabel("Percentage")
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(w1)))
plt.show()