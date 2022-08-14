import torch
import torch.nn as nn
import math
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
def cifar_model(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(32 * 8 * 8, 100),
        nn.ReLU(),
        linear_layer(100, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model


def cifar_model_large(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(64 * 8 * 8, 512),
        nn.ReLU(),
        linear_layer(512, 512),
        nn.ReLU(),
        linear_layer(512, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model
