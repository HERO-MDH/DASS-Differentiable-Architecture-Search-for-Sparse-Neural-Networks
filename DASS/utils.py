import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from layers import SubnetConv, SubnetLinear
import torch.nn as nn
import math
import sys
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  # state_dict = torch.load(model_path)
  # # create new ordererDict that does not contain 'module'
  # from collections import OrderedDict
  # new_state_dict = OrderedDict()
  # for k, v in state_dict.items():
  #   namekey = k[7:] # remove 'module'
  #   new_state_dict[namekey] = v
  # # load params
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

# TODO: avoid freezing bn_params
# Some utils are borrowed from https://github.com/allenai/hidden-networks
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores","arch_params"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores","arch_params"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
    

def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)
    

def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear
    else:
        raise ValueError("Incorrect layer type")


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def snip_init(model, criterion, optimizer, train_loader, device, args):
    print("Using SNIP initialization")
    assert args.exp_mode == "pretrain"
    optimizer.zero_grad()
    # init the score with kaiming normal init
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            nn.init.kaiming_normal_(m.popup_scores, mode="fan_in")

    set_prune_rate_model(model, 1.0)
    unfreeze_vars(model, "popup_scores")

    # take a forward pass and get gradients
    for _, data in enumerate(train_loader):
        images, target = data[0].to(device), data[1].to(device)

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        break

    # update scores with their respective connection sensitivty
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            print(m.popup_scores.data)
            m.popup_scores.data = m.popup_scores.grad.data.abs()
            print(m.popup_scores.data)

    # update k back to args.k.
    set_prune_rate_model(model, args.k)
    freeze_vars(model, "popup_scores")


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


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print(f"previous std = {torch.std(m.weight.data)}")
            m.weight.data = 1 / math.sqrt(k) * m.weight.data
            # print(f"new std = {torch.std(m.weight.data)}")


def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    set_prune_rate_model(model, args.k)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense


def dense_to_subnet(model, state_dict):
    """
        Load a dict with dense-layer in a model trained with subnet layers. 
    """
    model.load_state_dict(state_dict, strict=False)


def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    if verbose:
                        print(i, v, p)
        return np.mean(pl)


def sanity_check_paramter_updates(model, last_ckpt):
    """
        Check whether weigths/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        if hasattr(v, "weight") and hasattr(v, "popup_scores"):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2)
def get_id_popup_params(model):
    ids=[]
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            AA = list(map(id, m.popup_scores))
            for item in AA:
                ids.append(item)
    return  ids
def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd,)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
    return optim                
