import torch
import numpy as np 
import time
import torch
from torch.utils.data import dataloader
from get_models import get_model_darts, get_model_resnet18,get_model_moblienet
from cifar import CIFAR10
from layers import SubnetConv,SubnetLinear,prepare_model
from eval import base
import numpy as np 
import os
from layers import subnet_to_dense
from get_models import get_model_efficeint
# from Latency_compute import measure_latency_time
import torch.nn as nn
from statics import create_model,load_dense_model
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
def latency_event(model,dummy_input,repetitions):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(dummy_input)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)    
    return mean_syn,std_syn
def measure_latency_time(model, input_shape, is_cuda):
    INIT_TIMES = 100
    LAT_TIMES  = 1000
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
            tic = time.time()
            output = model(x)
            toc = time.time()
            lat.update(toc-tic, x.size(0))

    return lat.avg * 1000 # save as ms


def latency_nn_meter(model,predictor_name,predictor_version,input_shape):
    # import nni.retiarii.nn.pytorch as nn
    from nn_meter import load_latency_predictor
    predictor = load_latency_predictor(predictor_name,predictor_version)
    lat = predictor.predict(model, model_type='torch', input_shape=input_shape,apply_nni=False) 
    return lat


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
if __name__=="__main__":
    args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
    criterion = nn.CrossEntropyLoss()
    cl,ll =nn.Conv2d,nn.Linear
    data = CIFAR10(False)
    train_loader , test_loader = data.data_loaders()
    model = create_model(cl,ll,'darts_three_step',30,15,10)
    # prepare_model(model,args)
    # print(measure_latency_time(model,(1,3,32,32),False))
    hardware_name = 'myriadvpu_openvino2019r2'
    hardware_predictor_version = 1.0
    input_shape = (1,3,32,32)
    print(latency_nn_meter(model,hardware_name,hardware_predictor_version,input_shape))
