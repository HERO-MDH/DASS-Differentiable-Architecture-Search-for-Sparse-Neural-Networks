import imp
import os
from sched import scheduler
import sys
import time
import glob
import numpy as np
import torch
import utils
import genotypes
import logging
import argparse
import torch.nn as nn
from pruning import prepare_model_all,prepare_model_finetune,prepare_model_prune,prepare_model_pretrain
from pruning import initialize_scaled_score,set_init_popup,initialize_scores
from schedules import get_lr_policy
import torch.utils
import torch.nn.functional as F
from scipy.stats import norm
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from adv import pgd_whitebox, fgsm
from utils import get_layers,set_prune_rate_model,get_id_popup_params
from utils import unfreeze_vars,freeze_vars,initialize_scaled_score
from symbolic_interval.symbolic_network import sym_interval_analyze, naive_interval_analyze
from crown.eps_scheduler import EpsilonScheduler
from crown.bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel, Flatten
from model_search import Network
from model import NetworkCIFAR 
from architect import Architect
from adv import  trades_loss
# from pruning import Pruninig
import copy
from symbolic_interval.symbolic_network import sym_interval_analyze, naive_interval_analyze, mix_interval_analyze
from utils import AvgrageMeter
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id, split with ","')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup_epochs ')
##############################   PR-DARTS   ############################################################
parser.add_argument('--layer_type', type=str, default="subnet", help='type of layer')
parser.add_argument('--pruning_ratio', type=float, default=0.01, help='pruning ratio')
parser.add_argument('--pretrain_learning_rate', type=float, default=0.025, help='learning rate for pruning encoding')
parser.add_argument('--pretrain_weight_decay', type=float, default=1e-3, help='weight decay for pruning encoding')
parser.add_argument('--prune_learning_rate', type=float, default=0.1, help='learning rate for pruning encoding')
parser.add_argument('--prune_weight_decay', type=float, default=1e-3, help='weight decay for pruning encoding')
parser.add_argument('--freeze_bn', type=bool, default=False, help='freeze_bn parameters')
parser.add_argument('--init_type', type=str, default='kaiming_normal', help='init type name')
parser.add_argument('--lr_schedule', type=str, default='cosine', help='lr_ schedule name')
parser.add_argument('--finetune_learning_rate', type=float, default=0.025, help='learning rate for pruning encoding')
parser.add_argument('--finetune_weight_decay', type=float, default=1e-3, help='weight decay for pruning encoding')
parser.add_argument('--trainer', type=str, default="smooth", help='trainer')
parser.add_argument('--val_method', type=str, default="smooth", help='val_method')
###################################### Random smoothing   ######################################################
parser.add_argument('--noise_std', type=float, default=0.25, help='noise std')
parser.add_argument('--beta_smooth', type=float, default=6.0, help='beta trade off')
###############  Mix Train ##################################################################################
parser.add_argument('--epsilon_mix', type=float, default=0.007, help='epsilon_mix')
parser.add_argument('--schedule_length_mix', type=int, default=10, help='schedule_length_mix')
parser.add_argument('--starting_epsilon_mix', type=int, default=0, help='starting_epsilon_mix')
parser.add_argument('--schedule_start_mix', type=int, default=0, help='schedule_start_mix')
parser.add_argument('--interval_weight_mix', type=float, default=0.1, help='interval_weight_mix')
parser.add_argument('--mixtraink', type=float, default=0.1, help='mixtraink')

######################### crown _ ibp #####################333
parser.add_argument('--epsilon_ibp', type=float, default=0.007, help='epsilon_ibp')
parser.add_argument('--schedule_length_ibp', type=int, default=60, help='schedule_length_ibp')
parser.add_argument('--starting_epsilon_ibp', type=int, default=0, help='starting_epsilon_ibp')
parser.add_argument('--schedule_start_ibp', type=int, default=0, help='schedule_start_ibp')
############################## adv training ##############################3
parser.add_argument('--epsilon_adv', type=float, default=0.031, help='epsilon_adv')
parser.add_argument('--num_steps', type=int, default=10, help='num_steps')
parser.add_argument('--step_size', type=float, default=0.0078, help='step_size')
parser.add_argument('--clip_min', type=int, default=0, help='clip_min')
parser.add_argument('--clip_max', type=int, default=1, help='clip_max')
parser.add_argument('--const_init', type=bool, default=False, help='const_init')
parser.add_argument('--distance', type=str, default="l_inf", help='distance')
parser.add_argument('--beta', type=float, default=6.0, help='beta')
############################################################################################


args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  gpus = [int(i) for i in args.gpu.split(',')]
  if len(gpus) == 1:
    torch.cuda.set_device(int(args.gpu))
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  cl,ll = get_layers(args.layer_type)

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion,conv_layer=cl,linear_layer=ll)
  model = model.cuda()
  if len(gpus)>1:
    print("True")
    model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.module

  arch_params = list(map(id, model.arch_parameters()))
  popup_params = get_id_popup_params(model)
  weight_params = filter(lambda p: id(p) not in arch_params and id(p) not in popup_params,
                           model.parameters())
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  ############ PRETRAIN ###############
  prepare_model_all(model,args)
  set_init_popup(model,"kaiming_normal")
  set_prune_rate_model(model,1.0)
  architect = Architect(model, criterion, args)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(),
            lr=args.pretrain_learning_rate, weight_decay=args.pretrain_weight_decay)
  #lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  for epoch in range(args.epochs):#args.epochs
    # lr = lr_policy(epoch)
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    
    
    if args.trainer=='base':
      train_acc, train_obj = train_base(train_queue, valid_queue, model, architect, criterion, optimizer, lr,1.0,'pretrain',epoch)
    elif args.trainer=='smooth':
      train_acc, train_obj = train_random_smooth (train_queue, valid_queue, model, architect, criterion, optimizer, lr,1.0,'pretrain',epoch)
    elif args.trainer == 'mix':
      train_acc, train_obj = train_mix_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,1.0,'pretrain',epoch)
    elif args.trainer == 'crown': 
      train_acc, train_obj = train_cown_ibp (train_queue, valid_queue, model, architect, criterion, optimizer, lr,1.0,'pretrain',epoch)
    elif args.trainer == 'adv': 
       train_acc, train_obj =  Adv_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,1.0,'pretrain',epoch)
    logging.info('train_acc %f', train_acc)
    with torch.no_grad():
      if args.trainer=='base':
        valid_acc, valid_obj = infer_base(valid_queue, model, criterion)
      elif args.trainer=='smooth':
        valid_acc, valid_obj = infer_smooth(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'mix':
        valid_acc, valid_obj = infer_mixtrain(valid_queue, model, criterion, args)
      elif args.trainer == 'crown': 
        valid_acc, valid_obj = infer_ibp(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'adv':  
        valid_acc, valid_obj = infer_adv(valid_queue, model, criterion, args)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights_main_darts.pt'))


  ####### PRUNE ################### 
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(),
            lr=args.prune_learning_rate, weight_decay=args.prune_weight_decay)
  # lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  set_prune_rate_model(model,args.pruning_ratio)
  initialize_scaled_score(model)
  for epoch in range(50):
    # lr =lr_policy(epoch)
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    if args.trainer=='base':
      train_acc, train_obj = train_base(train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'prune',epoch)
    elif args.trainer=='smooth':
      train_acc, train_obj = train_random_smooth (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'prune',epoch)
    elif args.trainer == 'mix':
      train_acc, train_obj = train_mix_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'prune',epoch)
    elif args.trainer == 'crown': 
      train_acc, train_obj = train_cown_ibp (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'prune',epoch)
    elif args.trainer == 'adv': 
       train_acc, train_obj =  Adv_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'prune',epoch)
    logging.info('train_acc %f', train_acc)
    with torch.no_grad():
      if args.trainer=='base':
        valid_acc, valid_obj = infer_base(valid_queue, model, criterion)
      elif args.trainer=='smooth':
        valid_acc, valid_obj = infer_smooth(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'mix':
        valid_acc, valid_obj = infer_mixtrain(valid_queue, model, criterion, args)
      elif args.trainer == 'crown': 
        valid_acc, valid_obj = infer_ibp(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'adv':  
        valid_acc, valid_obj = infer_adv(valid_queue, model, criterion, args)
    logging.info('valid_acc %f', valid_acc)
    utils.save(model, os.path.join(args.save, 'weights_prune_darts.pt'))
  
  
  ############ FINETUNE ######################### 

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(),
            lr=args.finetune_learning_rate, weight_decay=args.finetune_weight_decay)
  # lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  set_prune_rate_model(model,args.pruning_ratio)
  for epoch in range(args.epochs):#args.epochs
    # lr = lr_policy(epoch)
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    if args.trainer=='base':
      train_acc, train_obj = train_base(train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'finetune',epoch)
    elif args.trainer=='smooth':
      train_acc, train_obj = train_random_smooth (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'finetune',epoch)
    elif args.trainer == 'mix':
      train_acc, train_obj = train_mix_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'finetune',epoch)
    elif args.trainer == 'crown': 
      train_acc, train_obj = train_cown_ibp (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'finetune',epoch)
    elif args.trainer == 'adv': 
       train_acc, train_obj =  Adv_train (train_queue, valid_queue, model, architect, criterion, optimizer, lr,args.pruning_ratio,'finetune',epoch)
    logging.info('train_acc %f', train_acc)
    with torch.no_grad():
      if args.trainer=='base':
        valid_acc, valid_obj = infer_base(valid_queue, model, criterion)
      elif args.trainer=='smooth':
        valid_acc, valid_obj = infer_smooth(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'mix':
        valid_acc, valid_obj = infer_mixtrain(valid_queue, model, criterion, args)
      elif args.trainer == 'crown': 
        valid_acc, valid_obj = infer_ibp(model, 'cuda', valid_queue, criterion, args)
      elif args.trainer == 'adv':  
        valid_acc, valid_obj = infer_adv(valid_queue, model, criterion, args)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights_fine_tune_darts.pt'))
   
##################################   Base Training       #################################    
def train_base(train_queue, valid_queue, model, architect, criterion, optimizer, lr,prun_ratio,name,epoch):
  batch_time = utils.AvgrageMeter()
  data_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,pr = prun_ratio)
    if name=="pretrain":
      model = prepare_model_pretrain(model,args)
    elif name=="prune":
      model= prepare_model_prune(model,args) 
    elif name=='finetune':
      model = prepare_model_finetune(model ,args) 
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    losses.update(loss.item(), n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    model = prepare_model_all(model,args)
  return top1.avg, objs.avg

###################################  Randomise smoothing training        ############################################3
def train_random_smooth(train_queue, valid_queue, model, architect, criterion, optimizer, lr,prun_ratio,name,epoch):
  batch_time = utils.AvgrageMeter()
  data_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
 
  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
     
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    if name == "pretrain" or name=="finetune":
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda()
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,pr = prun_ratio)
    if name=="pretrain":
      prepare_model_pretrain(model,args)
    elif name=="prune":
      prepare_model_prune(model,args) 
    elif name=='finetune':
      prepare_model_finetune(model ,args) 
    logits = model(input)
    loss_natural = nn.CrossEntropyLoss()(logits, target)
    loss_robust = (1.0 / len(input)) * nn.KLDivLoss(size_average=False)(
                F.log_softmax(
                    model(
                        input + torch.randn_like(input).to('cuda') * args.noise_std
                    ),
                    dim=1,
                ),
                F.softmax(logits, dim=1),
            )
    loss = loss_natural + args.beta_smooth * loss_robust
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    losses.update(loss.item(),n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    optimizer.zero_grad()
    
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    prepare_model_all(model,args)
 
  return top1.avg, objs.avg

#####################################################################################3
def set_epsilon(args, epoch):
    if epoch<args.schedule_length_mix:
        epsilon = epoch*(args.epsilon_mix - args.starting_epsilon_mix)/\
                args.schedule_length_mix + args.starting_epsilon_mix
    else:
        epsilon = args.epsilon_mix
    return epsilon

def set_interval_weight(args, epoch):
    interval_weight = args.interval_weight_mix * (2.5 **\
                ((max((epoch-args.schedule_length_mix), 0) // 5)))
    interval_weight = min(interval_weight, 50)
    return interval_weight


#################################### Mix Train  ######################################
def train_mix_train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,prun_ratio,name,epoch):
  epsilon = set_epsilon(args, epoch)
  k = args.mixtraink
  alpha = 0.8
  iw = set_interval_weight(args, epoch)

  print(
        " ->->->->->->->->->-> One epoch with MixTrain{} (SYM {:.3f})"
        " <-<-<-<-<-<-<-<-<-<-".format(k, epsilon)
    )

  batch_time = utils.AvgrageMeter()
  data_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  sym_losses = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  sym1 = utils.AvgrageMeter()
  
  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,pr = prun_ratio)
    if name=="pretrain":
      model = prepare_model_pretrain(model,args)
    elif name=="prune":
      model= prepare_model_prune(model,args) 
    elif name=='finetune':
      model = prepare_model_finetune(model ,args) 
    logits = model(input)
    ce = nn.CrossEntropyLoss()(logits, target)
    if(np.random.uniform()<=alpha):
      r = np.random.randint(low=0, high=input.shape[0], size=k)
      rce, rerr = sym_interval_analyze(model, epsilon, 
                            input[r], target[r],
                            use_cuda=torch.cuda.is_available(),
                            parallel=False)
      #print("sym:", rce.item(), ce.item())
      loss = iw * rce + ce
      sym_losses.update(rce.item(), k)
      sym1.update((1-rerr)*100., input.size(0))
    else:
      loss = ce

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    losses.update(ce.item(), n)
    optimizer.zero_grad()
    

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    model = prepare_model_all(model,args)
  return top1.avg, objs.avg
########################################## CROWN_IBP #######################
def train_cown_ibp(train_queue, valid_queue, model, architect, criterion, optimizer, lr,prun_ratio,name,epoch):
  num_class = 10
  sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
  for i in range(sa.shape[0]):
    for j in range(sa.shape[1]):
      if j < i:
          sa[i][j] = j
      else:
          sa[i][j] = j + 1
  sa = torch.LongTensor(sa) 
  batch_size = args.batch_size*2

  schedule_start = 0
  num_steps_per_epoch = len(train_queue)
  eps_scheduler = EpsilonScheduler("linear",
                args.schedule_start_ibp,
                ((args.schedule_start_ibp + args.schedule_length_ibp) - 1) *\
                num_steps_per_epoch, args.starting_epsilon_ibp,
                args.epsilon_ibp,
                num_steps_per_epoch)

  end_eps = eps_scheduler.get_eps(epoch+1, 0)
  start_eps = eps_scheduler.get_eps(epoch, 0)


  print(
        " ->->->->->->->->->-> One epoch with CROWN-IBP ({:.6f}-{:.6f})"
        " <-<-<-<-<-<-<-<-<-<-".format(start_eps, end_eps)
    )
  batch_time = utils.AvgrageMeter()
  data_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  ibp_losses = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  ibp_acc1 = utils.AvgrageMeter()
  

  model = BoundSequential.convert(model,\
                    {'same-slope': False, 'zero-lb': False,\
                    'one-lb': False}).to('cuda')

  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,pr = prun_ratio)
    if name=="pretrain":
      model = prepare_model_pretrain(model,args)
    elif name=="prune":
      model= prepare_model_prune(model,args) 
    elif name=='finetune':
      model = prepare_model_finetune(model ,args) 
    output = model(input, method_opt="forward")
    ce = nn.CrossEntropyLoss()(output, target)
    eps = eps_scheduler.get_eps(epoch, i) 
    # generate specifications
    c = torch.eye(num_class).type_as(input)[target].unsqueeze(1) -\
                torch.eye(num_class).type_as(input).unsqueeze(0) 
    # remove specifications to self
    I = (~(target.unsqueeze(1) ==\
            torch.arange(num_class).to('cuda').type_as(target).unsqueeze(0)))
    c = (c[I].view(input.size(0),num_class-1,num_class)).to('cuda')
    # scatter matrix to avoid compute margin to self
    sa_labels = sa[target].to('cuda')
    # storing computed lower bounds after scatter
    lb_s = torch.zeros(input.size(0), num_class).to('cuda')
    ub_s = torch.zeros(input.size(0), num_class).to('cuda')

    data_ub = torch.min(input + eps, input.max()).to('cuda')
    data_lb = torch.max(input - eps, input.min()).to('cuda')

    ub, ilb, relu_activity, unstable, dead, alive =\
                model(norm=np.inf, x_U=data_ub, x_L=data_lb,\
                eps=eps, C=c, method_opt="interval_range")

    crown_final_beta = 0.
    beta = (args.epsilon_ibp - eps * (1.0 - crown_final_beta)) / args.epsilon_ibp

    if beta < 1e-5:
      # print("pure naive")
      lb = ilb
    else:
      # print("crown-ibp")
      # get the CROWN bound using interval bounds 
      _, _, clb, bias = model(norm=np.inf, x_U=data_ub,\
                        x_L=data_lb, eps=eps, C=c,\
                        method_opt="backward_range")
      # how much better is crown-ibp better than ibp?
      # diff = (clb - ilb).sum().item()
      lb = clb * beta + ilb * (1 - beta)

    lb = lb_s.scatter(1, sa_labels, lb)
    robust_ce = criterion(-lb, target)

    #print(ce, robust_ce)
    racc = utils.accuracy(-lb, target, topk=(1,))

    loss = robust_ce
    prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    losses.update(ce.item(), n)
    ibp_losses.update(robust_ce.item(), n)
    ibp_acc1.update(racc[0].item(), n)
    optimizer.zero_grad()
    

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    model = prepare_model_all(model,args)
  return top1.avg, objs.avg
##########################################################################

##########################  Adv Training ##############################3
def Adv_train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,prun_ratio,name,epoch):
  batch_time = utils.AvgrageMeter()
  data_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,pr = prun_ratio)
    if name=="pretrain":
      model = prepare_model_pretrain(model,args)
    elif name=="prune":
      model= prepare_model_prune(model,args) 
    elif name=='finetune':
      model = prepare_model_finetune(model ,args) 
    output = model(input)
    loss = trades_loss(
            model=model,
            x_natural=input,
            y=target,
            device='cuda',
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon_adv,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            distance=args.distance,
        )


    prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    losses.update(loss.item(), n)
    
    optimizer.zero_grad()
    

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    model = prepare_model_all(model,args)
  return top1.avg, objs.avg


#################################  infer_base     #########################################
def infer_base(valid_queue, model, criterion):
  batch_time = utils.AvgrageMeter()
  losses = utils.AvgrageMeter()
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  
  

  model.eval()
  with torch.no_grad():
    end = time.time()
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      loss.update(loss.item(),n)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
      batch_time.update(time.time() - end)
      end = time.time()
      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg
######################################## infer_adv #############################
def infer_adv(valid_queue, model, criterion, args):
    
    batch_time = utils.AvgrageMeter()
    losses = utils.AvgrageMeter()
    adv_losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    adv_top1 = utils.AvgrageMeter()
    adv_top5 = utils.AvgrageMeter()
  

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(valid_queue):
            images, target = data[0].to('cuda'), data[1].to('cuda')

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # adversarial images
            images = pgd_whitebox(
                model,
                images,
                target,
                'cuda',
                args.epsilon_adv,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=not args.const_init,
            )

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1.item(), images.size(0))
            adv_top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.report_freq == 0:
                logging.info('valid %03d %e %f %f %e %f %f %f', i, losses.avg, top1.avg, top5.avg, adv_losses.avg, adv_top1.avg, adv_top5.avg,batch_time.avg)
           

    return adv_top1.avg, adv_top5.avg
################################ infer_mixtrain ###################333333
def infer_mixtrain(valid_queue, model, criterion, args):
    batch_time = utils.AvgrageMeter()
    losses = utils.AvgrageMeter()
    sym_losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    sym_top1 = utils.AvgrageMeter()
  

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(valid_queue):
            images, target = data[0].to('cuda'), data[1].to('cuda')

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            rce_avg = 0
            rerr_avg = 0
            for r in range(images.shape[0]):

                rce, rerr = sym_interval_analyze(
                    model,
                    args.epsilon_mix,
                    images[r : r + 1],
                    target[r : r + 1],
                    use_cuda=torch.cuda.is_available(),
                    parallel=False,
                )
                rce_avg = rce_avg + rce.item()
                rerr_avg = rerr_avg + rerr

            rce_avg = rce_avg / float(images.shape[0])
            rerr_avg = rerr_avg / float(images.shape[0])

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            sym_losses.update(rce_avg, images.size(0))
            sym_top1.update((1 - rerr_avg) * 100.0, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.report_freq == 0:
                logging.info('valid %03d %e %f %f %e %f %f', i, losses.avg, top1.avg, top5.avg, sym_losses.avg, sym_top1.avg, batch_time.avg)
            

    return sym_top1.avg, sym_top1.avg
###################################### infer _ ibp ##################3
def infer_ibp(model, device, val_loader, criterion, args):
    batch_time = utils.AvgrageMeter()
    losses = utils.AvgrageMeter()
    ibp_losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    ibp_top1 = utils.AvgrageMeter()

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            rce, rerr = naive_interval_analyze(
                model,
                args.epsilon_ibp,
                images,
                target,
                use_cuda=torch.cuda.is_available(),
                parallel=False,
            )

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            ibp_losses.update(rce.item(), images.size(0))
            ibp_top1.update((1 - rerr) * 100.0, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.report_freq == 0:
                logging.info('valid %03d %e %f %f %e %f %f', i,losses.avg, top1.avg,top5.avg, ibp_losses.avg, ibp_top1.avg, batch_time.avg)


    return ibp_top1.avg, ibp_top1.avg

################################## infer_ smooth   ################################
def infer_smooth(model, device, val_loader, criterion, args):
    batch_time = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    rad = utils.AvgrageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # Defult: evaluate on 10 random samples of additive gaussian noise.
            output = []
            for _ in range(10):
                # add noise
                noise = torch.randn_like(images).to(device) * args.noise_std

                output.append(F.softmax(model(images + noise), -1))

            output = torch.sum(torch.stack(output), axis=0)/10

            p_max, _ = output.max(dim=-1)
            radii = (args.noise_std + 1e-16) * norm.ppf(p_max.data.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            rad.update(np.mean(radii))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.report_freq == 0:
                logging.info('valid %03d %e %f %f %f', i, rad.avg, top1.avg, top5.avg,batch_time.avg)
            
           
    return top1.avg, rad.avg
if __name__ == '__main__':
  main()

