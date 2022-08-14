import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pruning import prepare_model_all,set_prune_rate_model
def unfreeze_vars(model, var_name):
    assert var_name in ["weight","bias","arch_params", "weight_params", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True




def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, criterion, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.scores_init_type = args.init_type
    self.model = model
    print(self.model.conv_layer,"**********************")
    self.criterion = criterion
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    #self.optimizer = nn.DataParallel(self.optimizer, device_ids=[0,1])
    self.args = args
  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    logits = self.model(input)
    loss_natural = self.criterion(logits, target)
    loss_robust = (1.0 / len(input)) * nn.KLDivLoss(size_average=False)(
                F.log_softmax(
                    self.model(
                        input + torch.randn_like(input).to('cuda') * self.args.noise_std
                    ),
                    dim=1,
                ),
                F.softmax(logits, dim=1),
            )
    # loss = self.model._loss(input, target)
    loss = loss_natural #+ self.args.beta_smooth * loss_robust
    theta = _concat(self.model.parameters()).detach()
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).detach() + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled,pr):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer,pr)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    logits = self.model(input_valid)
    loss = self.criterion(logits, target_valid)
    # loss = self.model._loss(input_valid, target_valid)
    loss.backward()
  
  
  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,pr):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    prepare_model_all(unrolled_model,self.args)
    set_prune_rate_model(unrolled_model,pr)
    
    unrolled_logits = unrolled_model(input_valid)
    unrolled_loss = self.criterion(unrolled_logits, target_valid)
    # unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.detach() for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = g.data
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    logits = self.model(input)
    loss = self.criterion(logits, target)
    # loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    logits = self.model(input)
    loss = self.criterion(logits, target)
    # loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

