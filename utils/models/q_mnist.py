import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ..quantization_utils.quant_modules import QuantAct, QuantConv2d, QuantDropout, QuantLinear


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Q_MNIST(torch.nn.Module):
    """
    Quantized Neural Network model for MNIST dataset.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point network.
    weight_precision : int, default 8
        Bitwidth for quantized weights.
    bias_precision : int, default 32
        Bitwidth for quantized bias.
    act_precision : int, default 16
        Bitwidth for quantized activations.
    """
    def __init__(self, model,
                       weight_precision=8,
                       bias_precision=32,
                       act_precision=16) -> None:
        super().__init__()
        if model is None:
            raise ValueError('Model cannot be None')
        
        self.quant_input = QuantAct(act_precision)
        self.quant_act1 = QuantAct(act_precision)

        layer = getattr(model, 'conv1')
        quant_layer = QuantConv2d()
        quant_layer.set_param(layer)
        setattr(self, 'conv1', quant_layer)

        layer = getattr(model, 'conv2')
        quant_layer = QuantConv2d()
        quant_layer.set_param(layer)
        setattr(self, 'conv2', quant_layer)

        layer = getattr(model, 'relu')
        setattr(self, 'relu', layer)

        layer = getattr(model, 'fc1')
        quant_layer = QuantLinear(weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc1', quant_layer)

        layer = getattr(model, 'fc2')
        quant_layer = QuantLinear(weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc2', quant_layer)
    
    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)
        # quantized conv 
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(self.relu(x), act_scaling_factor, weight_scaling_factor)
        # quantized conv 
        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(self.relu(x), act_scaling_factor, weight_scaling_factor)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        # fully connected layers 
        x, weight_scaling_factor = self.fc1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(self.relu(x), act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.fc2(x, act_scaling_factor)
        self.relu(x)
        
        output = F.log_softmax(x, dim=1)
        return output


def q_mnist(model=None):
    if model is None:
        model = MNIST()
    return Q_MNIST(model)
