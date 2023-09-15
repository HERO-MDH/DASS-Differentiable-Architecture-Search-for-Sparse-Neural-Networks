# DASS: Differentiable Architecture Search for Sparse Neural Networks

This repository contains code and trained models for the DASS paper [DASS: Differentiable Architecture Search for Sparse Neural Networks](https://dl.acm.org/doi/10.1145/3609385).
DASS is a method for searching architectures of a network with pruned weights.
When using the same pruning method, our searched architectures outperform other pruned networks whose architectures are well known floating-point networks. Our searched architectures also achieve competitive results with non-pruned networks.

## This repository

The contents of this repository are as follows:

* [PRDARTS/](PRDARTS) contains the code for search pruned network.
* [hydra/](hydra) contains the code for pruning method.
* [scripts/](scripts) contains utility functions to evaluate the resluts.
* [figures/](figures) contains the main figures of results.

## Requirements
```
Python >= 3.8.12, PyTorch == 1.8.1, torchvision == 0.9.1
```
All dependencies of python enviornment can be install by run:

```
pip install -r requirement.txt
```

## Datasets
CIFAR-10 can be automatically downloaded by torchvision, ImageNet needs to be manually downloaded following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Inference with Pre-Trained Models

To reproduce the results reported in the paper, you can use the pretrained models.

For CIFAR10 we evaulate PR-DARTS-Tiny (Small, Medium,Large) by the following setting:

- `--source_net`: pathe to the pretrain weights

Create the model by run:

```
model= create_model(cl,ll,'PR-DARTS architecture',InitChannels,Layers,10)
```

Run the following script to generate accuracy and latency and number of parameters:

```
cd Scipts && python statics.py 
```

## Pruned Architecture search (PR-DARTS)
To carry out pruned architecture search run:

```
cd PRDARTS && python train_search.py --unrolled   
```
Note the validation performance in this step does not indicate the final performance of the pruned network. One must use a pruning method to the obtained genotype/architecture from scratch. Also be aware that different runs would end up with different local minimum.

- `--layer_type`: select subnet for using PrundConv and PrundLinear 
- `--pruning_ratio`: float number for pruning ratio
- `--prune_learning_rate`: learning rate for pruning step
- `--finetune_learning_rate`: learning rate for finetuning step

## Pruning Method (HYDRA)

We will use `hydra/train.py` for all our experiments on the CIFAR-10. For ImageNet, we will use `hydra/train_imagenet.py`. It provides the flexibility to work with pre-training, pruning, and Finetuning steps.

- `--exp_mode`: select from pretrain, prune, finetune
- `--dataset`: cifar10, imagenet
- `--k`: pruning ratio

You should set the following parameters for PR-DARTS:

- `init_channels_td`: 30 for PR-DARTS-Tiny, 36 for PR-DARTS-Small, 86 for PR-DARTS-Medium and 108 for PR-DARTS-Large
- `layers_td`: 16 for PR-DARTS-Tiny, 20 for PR-DARTS-Small, 12 for PR-DARTS-Medium and 14 for PR-DARTS-Large

## Pre-training

In pre-training, we train the networks with `k=1` i.e, without pruning.

`cd hydra && python train.py --arch darts --exp-mode pretrain --configs configs/configs.yml `


## Pruning

In pruning steps following command will prune the pre-trained  network to 99% pruning ratio.

`python train.py --arch darts --exp-mode prune --configs configs.yml --k 0.01 --scaled-score-init --source-net pretrained_net_checkpoint_path --save-dense`

It used the proposed scaled initialization of prunning parameters. 


## Fine-tuning

In the fine-tuning step, we will update the non-pruned weights 

`python train.py --arch darts --exp-mode finetune --configs configs.yml --k 0.01 --source-net pruned_net_checkpoint_path --save-dense `




