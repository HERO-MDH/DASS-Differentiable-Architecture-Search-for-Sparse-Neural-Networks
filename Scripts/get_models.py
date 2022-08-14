
from cifar_model import cifar_model, cifar_model_large
from model import NetworkCIFAR
from model_imagenet import NetworkImageNet
from resnet_cifar import resnet18
from resnet import ResNet18
from mobilenet import mobilenet_v2
from effnetv2_image import EfficientNetV2 as EfficientNetV2_image
from mobilenetv2_image import mobilenet_v2_image
from effenetv2 import EfficientNetV2
import genotypes
from layers import SubnetConv,SubnetLinear
import numpy as np 
def get_model_darts(conv_layer,linear_layer,channels,layers,num_class,arch):
    genotype = eval("genotypes.%s" % arch)
    model = NetworkCIFAR(channels, num_class, layers, True , genotype,True,conv_layer = conv_layer,linear_layer = linear_layer)
    return model
def get_model_darts_imagenet(conv_layer,linear_layer,channels,layers,num_class,arch):
    genotype = eval("genotypes.%s" % arch)
    model = NetworkImageNet(channels, num_class, layers, True, genotype,conv_layer = conv_layer,linear_layer = linear_layer)
    return model   
def get_model_resnet18(conv_layer,linear_layer):
    model = resnet18(conv_layer,linear_layer,'kaiming_normal')
    return model    
def get_model_resnet18_imgenet(conv_layer,linear_layer):
    model = ResNet18(conv_layer,linear_layer,'kaiming_normal')
    return model
def get_model_moblienet(conv_layer,linear_layer):
    model = mobilenet_v2(conv_layer,linear_layer,'kaiming_normal',10)
    return model      
def get_model_moblienet_imagenet(conv_layer,linear_layer):
    model = mobilenet_v2_image(conv_layer,linear_layer,'kaiming_normal',1000)
    return model  
def get_model_efficeint(conv_layer,linear_layer):
    model = EfficientNetV2(conv_layer,linear_layer,'kaiming_normal',10)
    return model     
def get_model_efficeint_imagenet(conv_layer,linear_layer):
    model = EfficientNetV2_image(conv_layer,linear_layer,'kaiming_normal',1000)
    return model 
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


if __name__=="__main__":
    cl,ll = SubnetConv,SubnetLinear
    model = get_model_darts(SubnetConv,SubnetLinear,36,20,10,"DARTS_PRUNE_Randomizd_Smoothing")
    # model = get_model_resnet18_imgenet(SubnetConv,SubnetLinear)
    print(count_parameters_in_MB(model))
