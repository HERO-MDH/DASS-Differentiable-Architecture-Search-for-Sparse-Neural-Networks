import os
from re import A
import torch
import torch.nn as nn
from cifar import CIFAR10
import cv2
import random
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
import pandas as pd
from CKA import linear_CKA, kernel_CKA
import seaborn as sns
from scipy.stats import kendalltau
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from statics import create_model,load_dense_model,evaluate_model
# Loading the model
data = CIFAR10(False)
train_loader , test_loader = data.data_loaders()
###########################################################
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image   
def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
        print(indices)
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()

def visualize_tsne(tsne, labels):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
if __name__=="__main__":
    #############################################
    args={'exp_mode':'pretrain','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':1.0}
    criterion = nn.CrossEntropyLoss()
    cl,ll =nn.Conv2d,nn.Linear

    source_net_dense= "models/model_best_3for_dense_pretrain.pth.tar"
    model = create_model(cl,ll,'darts_three_step',36,20,10)
    model = load_dense_model(model,source_net_dense,'cpu',args)
    # top1,top5 = evaluate_model(model,'cpu',test_loader,criterion)
    # print(top1,top5)
    #########################################################
    colors_per_class = {
    0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
    5 : [128, 80, 128],
    6 : [87, 101, 116],
    7: [52, 31, 151],
    8 : [0, 0, 0],
    9 : [100, 100, 255],}

    
    ##############################################33
    args={'exp_mode':'finetune','freeze_bn':False,'scores_init_type':"kaiming_normal",'prune_ratio':0.01}
    criterion = nn.CrossEntropyLoss()
    cl,ll =nn.Conv2d,nn.Linear
    data = CIFAR10(False)
    source_net_dense= "models/model_best_3for_dense_finetune.pth.tar"
    model1 = create_model(cl,ll,'darts_three_step',36,20,10)
    model1 = load_dense_model(model1,source_net_dense,'cpu',args)
    model.eval()
    model1.eval()
    model.drop_path_prob = 0
    model1.drop_path_prob = 0
    activation = {}

    model.cells[2]._ops[7].register_forward_hook(get_activation('identity_12'))   
    model1.cells[2]._ops[7].register_forward_hook(get_activation('identity1_12'))  
    model = model.to('cpu')
    model1 = model1.to('cpu')
    num=0
    features1 = None
    features2 = None
    labels=None
    for data,label in test_loader:   
        num+=1
        if labels is not None:
            labels = np.concatenate((labels, label))
        else:
            labels = label   
        data=data.to('cpu')
        label = label.to('cpu')
        model.forward(data)
        output1 = activation['identity_12']
        current_outputs1 = output1.detach().numpy().reshape(-1,output1.shape[1]*output1.shape[2]*output1.shape[3])
        if features1 is not None:
            features1 = np.concatenate((features1, current_outputs1))
        else:
            features1 = current_outputs1    
        activation={}
        model1.forward(data)
        output2 = activation['identity1_12']
        print(output2.shape)
        current_outputs2 = output2.detach().numpy().reshape(-1,output2.shape[1]*output2.shape[2]*output2.shape[3])
        if features2 is not None:
            features2 = np.concatenate((features2, current_outputs2))
        else:
            features2 = current_outputs2
        if num==200:
            break
    print(features1.shape)
    print(features2.shape)
    print(labels)
    fix_random_seeds()    
    tsne1 = TSNE(n_components=2).fit_transform(features1)
    tsne2 = TSNE(n_components=2).fit_transform(features2)
    visualize_tsne(tsne1, labels)
