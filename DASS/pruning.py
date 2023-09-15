
import torch
import torch.nn as nn
import math
import numpy as np
import copy
def prepare_model_prune (model,args):
    unfreeze_vars(model,"popup_scores")
    freeze_vars( model,"weight", args.freeze_bn)
    freeze_vars(model,"bias", args.freeze_bn)
    freeze_vars(model,"arch_params")

def prepare_model_pretrain (model,args):
    freeze_vars(model,"popup_scores",args.freeze_bn)
    unfreeze_vars( model,"weight")
    unfreeze_vars(model,"bias")
    unfreeze_vars(model,"arch_params")
    
def prepare_model_finetune(model,args):
       
    freeze_vars(model,"popup_scores",args.freeze_bn)
    unfreeze_vars(model, "weight")
    unfreeze_vars(model,"bias")
    unfreeze_vars(model,"arch_params") 
    
def prepare_model_all(model,args):
    unfreeze_vars(model,"popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model,"bias")
    unfreeze_vars(model,"arch_params")  
  
def set_init_popup( model,scores_init_type):
    initialize_scores(model,scores_init_type)
    
def freeze_vars(  model, var_name, freeze_bn=False):
    assert var_name in ["weight", "bias", "popup_scores","arch_params"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False               
def unfreeze_vars( model,var_name):
    assert var_name in ["weight", "bias", "popup_scores","arch_params"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
               
def set_prune_rate_model(  model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate) 
           
def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )          
def initialize_scaled_score( model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in  model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )
         