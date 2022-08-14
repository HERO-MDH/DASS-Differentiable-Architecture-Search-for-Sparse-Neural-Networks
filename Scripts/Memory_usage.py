import torch
import torch.nn as nn
from cifar import CIFAR10
import psutil
import os
from statics import create_model,load_dense_model
args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
criterion = nn.CrossEntropyLoss()
cl,ll =nn.Conv2d,nn.Linear
data = CIFAR10(False)
source_net_dense = "models/model_best_ft_dense_DARTS_orig_120E.pth.tar"
train_loader , test_loader = data.data_loaders()
model = create_model(cl,ll,'darts_orig',36,20,10)
model = load_dense_model(model,source_net_dense,'cpu',args)
model.drop_path_prob = 0
x = torch.randn(2, 3, 32, 32)
process = psutil.Process(os.getpid())
for idx in range(100):
    print(idx, process.memory_full_info().rss / 1024**2)
    out = model(x)