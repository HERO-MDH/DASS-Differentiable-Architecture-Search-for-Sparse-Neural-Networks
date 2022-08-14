import os
import sys
import time
import glob
from models.utils import save
import numpy as np
import torch
import models.utils as utils 
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.nn as nn
from utils.model import initialize_scaled_score
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from trainer.base import train as train_popup
from models.model_search import Network
from models.model import NetworkCIFAR
from models.architect import Architect
import models.genotypes as genotypes
from utils.schedules import get_optimizer
from utils.adv import trades_loss
def get_id_popup_params(model):
    ids=[]
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            AA = list(map(id, m.popup_scores))
            for item in AA:
                ids.append(item)
    return  ids

def unfreeze_vars(model, var_name):
    assert var_name in ["weight","bias","arch_params", "weight_params", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores","arch_params", "weight_params"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False
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
def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)
def train_search(args,conv_layer,linear_layer):
    args.save_darts = 'search-{}-{}'.format(args.save_darts, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save_darts, scripts_to_save=glob.glob('models/*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_darts, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    CIFAR_CLASSES = 10
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    gpus = [int(i) for i in args.gpu.split(',')]
    if len(gpus) == 1:
        torch.cuda.set_device(int(args.gpu))
    # cudnn.benchmark = True
    torch.manual_seed(args.seed)
    # cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels_darts, CIFAR_CLASSES, args.layers_darts, criterion,conv_layer = conv_layer,linear_layer = linear_layer)
    model = model.cuda()
    if len(gpus) > 1:
        print("True")
        model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model = model.module
    set_prune_rate_model(model,args.search_k_darts)
    arch_params = list(map(id, model.arch_parameters()))
    popup_params = get_id_popup_params(model)
    weight_params = filter(lambda p: id(p) not in arch_params and id(p) not in popup_params,
                           model.parameters())
    unfreeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        # model.parameters(),
        weight_params,
        args.learning_rate_darts,
        momentum=args.momentum,
        weight_decay=args.weight_decay_darts)
    #optimizer = nn.DataParallel(optimizer, device_ids=gpus)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data_darts, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion_darts * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_darts,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_darts,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs_darts), eta_min=args.learning_rate_min_darts)

    architect = Architect(model, criterion, args)
    initialize_scaled_score(model)
    for epoch in range(args.epochs_darts):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,args,epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion,args)
        logging.info('valid_acc %f', valid_acc)

        save(model, os.path.join(args.save_darts, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,args,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled_darts)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_darts)
        optimizer.step()
        ##################popup params train###########################
        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight")
        freeze_vars(model, "bias")
        freeze_vars(model,'arch_params')
        optimizer = get_optimizer(model, args)
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        gpu_list = [int(i) for i in args.gpu.strip().split(",")]
        device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")
        # writer = SummaryWriter(os.path.join(args.save_darts, "tensorboard"))
        output= model(input)
        # loss = trades_loss(
        #     model=model,
        #     x_natural=input,
        #     y=target,
        #     device=device,
        #     optimizer=optimizer,
        #     step_size=args.step_size,
        #     epsilon=args.epsilon,
        #     perturb_steps=args.num_steps,
        #     beta=args.beta,
        #     clip_min=args.clip_min,
        #     clip_max=args.clip_max,
        #     distance=args.distance,
        # )
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ##########################################################
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        unfreeze_vars(model, "popup_scores")
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        
    return top1.avg, objs.avg
def get_arch(args,conv_layer,linear_layer):
    CIFAR_CLASSES = 1000
    if args.train_search:
        train_search(args, conv_layer, linear_layer)
    genotype = eval("genotypes.%s" % args.arch_td)
    model = NetworkCIFAR(args.init_channels_td, CIFAR_CLASSES, args.layers_td, args.auxiliary_td, genotype,True,conv_layer = conv_layer,linear_layer = linear_layer)
    model = model.cuda()
    return model



def darts(conv_layer, linear_layer, init_type , args, **kwargs):
    return get_arch(args,conv_layer,linear_layer)