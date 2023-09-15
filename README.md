# DASS: Differentiable Architecture Search for Sparse Neural Networks

This repository contains code and trained models for the DASS paper [DASS: Differentiable Architecture Search for Sparse Neural Networks](https://dl.acm.org/doi/10.1145/3609385).
DASS proposes a new method to search for sparsity-friendly neural architectures. It is done by adding two new sparse operations to the search space and modifying the search objective. We propose two novel parametric SparseConv and SparseLinear operations in order to expand the search space to include sparse operations. In particular, these operations create a flexible search space due to their use of sparse parametric versions of linear and convolutional operations. The proposed search objective lets us train the architecture based on the sparsity of the search space operations.

## This repository

The contents of this repository are as follows:

* [DASS/](DASS) contains the code to search sparse networks.
* [hydra/](hydra) contains the code for the pruning method.
* [scripts/](scripts) contains utility functions to evaluate the results.
* [figures/](figures) contains the main figures of the results.

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

To reproduce the results reported in the paper, you can use the pre-trained models.

For CIFAR10, we evaluate DASS-Tiny (Small, Medium, large) by the following setting:

- `--source_net`: path to the pre-train weights

Create the model by running:

```
model = create_model(cl,ll,'DASS architecture',InitChannels,Layers,10)
```

Run the following script to generate accuracy, latency, and the number of parameters:

```
cd Scipts && python statics.py 
```

## DASS: Differentiable Architecture Search for Sparse Neural Networks
To run the search algorithm:

```
cd DASS && python train_search.py --unrolled   
```
Note that the validation performance in this step does not indicate the final performance of the pruned network. One must use a pruning method to obtain the genotype or architecture from scratch. Also, be aware that different runs would end up with different local minimums.

- `--layer_type`: select a subnet for using PrundConv and PrundLinear 
- `--pruning_ratio`: float number for pruning ratio
- `--prune_learning_rate`: learning rate for pruning step
- `--finetune_learning_rate`: learning rate for finetuning step

## Pruning Method (HYDRA)

We will use `hydra/train.py` for all our experiments on the CIFAR-10. For ImageNet, we will use `hydra/train_imagenet.py`. It provides the flexibility to work with pre-training, pruning, and fine-tuning steps.

- `--exp_mode`: select from pre-train, prune, finetune
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




