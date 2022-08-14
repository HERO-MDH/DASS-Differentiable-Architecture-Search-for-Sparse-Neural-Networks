import torch
from torch.utils.data import dataloader
from get_models import get_model_darts, get_model_resnet18,get_model_moblienet,get_cifar_model
from cifar import CIFAR10
from layers import SubnetConv,SubnetLinear,prepare_model
from eval import base
import numpy as np 
import time
from get_models import get_model_efficeint
import torch.nn as nn
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
INIT_TIMES = 100
LAT_TIMES  = 1000
args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
criterion = nn.CrossEntropyLoss()
cl,ll = SubnetConv,SubnetLinear
data = CIFAR10(False)
train_loader , test_loader = data.data_loaders()
model = get_model_moblienet(cl,ll)

prepare_model(model,args)
source_net = "model_best_ft_mobilenet.pth.tar"
import os
if os.path.isfile(source_net):
    checkpoint = torch.load(source_net, map_location='cuda')
    model.load_state_dict(checkpoint["state_dict"])
else:
    print('no') 



def measure_latency_in_ms_event(model, input_shape, is_cuda):
    start = torch.cuda.Event(enable_timing=True)    
    end = torch.cuda.Event(enable_timing=True) 
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            start.record()
            output = model(x)
            end.record()
            torch.cuda.synchronize()
            lat.update(start.elapsed_time(end), x.size(0))

    return lat.avg * 1000 # save as ms


def measure_latency_in_ms_profiler(model, input_shape, is_cuda):
     
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            with torch.autograd.profiler.profile(use_cuda=True) as prof:   
                output = model(x)
                torch.cuda.synchronize()
                lat.update(prof., x.size(0))

    return lat.avg * 1000 # save as ms





class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


