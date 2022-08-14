import torch
from torch.utils.data import dataloader
from get_models import get_model_darts, get_model_efficeint_imagenet, get_model_resnet18,get_model_moblienet
from get_models import get_model_darts_imagenet,get_model_efficeint_imagenet
from get_models import get_model_moblienet_imagenet,get_model_resnet18_imgenet
from cifar import CIFAR10
from layers import SubnetConv,SubnetLinear,prepare_model
from eval import base
import numpy as np 
import os
from layers import subnet_to_dense
from get_models import get_model_efficeint
# from Latency_compute import measure_latency_time
import torch.nn as nn
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
def create_model(conv_layer,linear_layer,model_name,channels,layers,num_class):
  
  if model_name=='darts_orig':
    model = get_model_darts(conv_layer,linear_layer,channels,layers,num_class,"DARTS_ORIG_120")
  elif model_name=='darts_three_step':
    model = get_model_darts(conv_layer,linear_layer,channels,layers,num_class,"DARTS_prune_three_for")
  elif model_name=='darts_three_step_90':
    model = get_model_darts(conv_layer,linear_layer,channels,layers,num_class,"DARTS_prune_three_for_90")
  elif model_name=='darts_three_step_95':
    model = get_model_darts(conv_layer,linear_layer,channels,layers,num_class,"DARTS_prune_three_for_95")  
  elif model_name=="resnet18":
    model = get_model_resnet18(conv_layer,linear_layer)
  elif model_name=="mobilenetv2":
    model = get_model_moblienet(conv_layer,linear_layer)
  elif model_name=="efficientnet":
    model = get_model_efficeint(conv_layer,linear_layer)     
  return model
def create_model_imagenet(conv_layer,linear_layer,model_name,channels,layers,num_class):
  if model_name=='darts_orig':
    model = get_model_darts_imagenet(conv_layer,linear_layer,channels,layers,num_class,"DARTS_ORIG_120")
  elif model_name=='darts_three_step':
    model = get_model_darts_imagenet(conv_layer,linear_layer,channels,layers,num_class,"DARTS_prune_three_for")
  elif model_name=="resnet18":
    model = get_model_resnet18_imgenet(conv_layer,linear_layer)
  elif model_name=="mobilenetv2":
    model = get_model_moblienet_imagenet(conv_layer,linear_layer)
  elif model_name=="efficientnet":
    model = get_model_efficeint_imagenet(conv_layer,linear_layer)     
  return model
def load_model(args,model,source_net,device):
  prepare_model(model,args)
  if os.path.isfile(source_net):
    checkpoint = torch.load(source_net, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
  else:
    print('no model')   
  return model

def evaluate_model(model,device,test_loader,criterion):
  top1,top5 = base(model,device,test_loader,criterion,False)
  return top1,top5
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
# def flops(model):
#     macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
#                                             print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def create_dense_model(model,k,name):
  state_dict = model.state_dict()
  torch.save(
            subnet_to_dense(state_dict, k),
            os.path.join(name),
        )
def load_dense_model(model,source_net_dense,device,args):
  prepare_model(model,args)
  if os.path.isfile(source_net_dense):
    sate_dict = torch.load(source_net_dense, map_location=device)
    model.load_state_dict(sate_dict)
  else:
    print('no model')   
  return model
if __name__ =="__main__":
  args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
  
  criterion = nn.CrossEntropyLoss()
  cl,ll =SubnetConv,SubnetLinear
  data = CIFAR10(False)
  source_net = "models/model_best_ft_darts_orig.pth.tar"
  # source_net_dense = "models/model_best_ft_dense_darts_12_86.pth.tar"
  train_loader , test_loader = data.data_loaders()

  model = create_model(cl,ll,'darts_three_step',36,20,10)
  model = load_model(args,model,source_net,'cpu')
  prepare_model(model,args)
  # model = load_dense_model(model,source_net_dense,'cpu',args)
  top1,top5 = evaluate_model(model,'cpu',test_loader,criterion)
  print(top1,top5)
  print(count_parameters_in_MB(model))
  print(measure_latency_time(model,(1,3,32,32),False))
  # create_dense_model(model,0.01,source_net_dense)
  
