"""
    3 Hidden Layer Jet Tagging Model
"""

import torch.nn as nn
from ..quantization_utils.quant_modules import QuantAct, QuantLinear


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        """Multi-Layer Perceptron"""
        super(MultiLayerPerceptron, self).__init__()
  
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        softmax_out = self.softmax(self.fc4(x))
        return softmax_out


class Q_JetTagger(nn.Module):
    """
    Quantized JetTagger model.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point JetTagger.
    weight_precision : int, default 8
        Bitwidth for quantized weights.
    bias_precision : int, default 32
        Bitwidth for quantized bias.
    act_precision : int, default 16
        Bitwidth for quantized activations.
    dense_out : bool, default False
        Return output of each dense layer with prediction.
    quant_out : bool, default False
        Return output of each quant layer with prediction.
    batchnorm : bool, default False
        Use batch normalization.
    silu : bool, default False
        Use SiLU activation function.
    gelu : bool, default False
        Use GELU activation function.
    """
    def __init__(self, 
                 model, 
                 weight_precision=8,
                 bias_precision=32,
                 act_precision=16,
                 dense_out=False,
                 quant_out=False,
                 batchnorm=False,
                 silu=False,
                 gelu=False):
        super(Q_JetTagger, self).__init__()
        self.dense_out = dense_out
        self.quant_out = quant_out
        self.batchnorm = batchnorm

        self.quant_input = QuantAct(act_precision)

        layer = getattr(model, 'fc1')
        quant_layer = QuantLinear(weight_bit=weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc1', quant_layer)
        self.quant_act1 = QuantAct(act_precision)

        layer = getattr(model, 'fc2')
        quant_layer = QuantLinear(weight_bit=weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc2', quant_layer)
        self.quant_act2 = QuantAct(act_precision)

        layer = getattr(model, 'fc3')
        quant_layer = QuantLinear(weight_bit=weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc3', quant_layer)
        self.quant_act3 = QuantAct(act_precision)

        layer = getattr(model, 'fc4')
        quant_layer = QuantLinear(weight_bit=weight_precision,bias_bit=bias_precision)
        quant_layer.set_param(layer)
        setattr(self, 'fc4', quant_layer)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(32)
            self.bn3 = nn.BatchNorm1d(32)

        self.act = nn.ReLU()
        if silu:
            self.act = nn.SiLU()
        if gelu:
            self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        q0 = x.detach().clone()

        x = self.fc1(x, act_scaling_factor)
        d1 = x.detach().clone()
        if self.batchnorm:
            x = self.bn1(x)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor)
        q1 = x.detach().clone()

        x = self.fc2(x, act_scaling_factor)
        d2 = x.detach().clone()
        if self.batchnorm:
            x = self.bn2(x)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor)
        q2 = x.detach().clone()

        x = self.fc3(x, act_scaling_factor)
        d3 = x.detach().clone()
        if self.batchnorm:
            x = self.bn3(x)
        x = self.act(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor)
        q3 = x.detach().clone()

        x = self.fc4(x, act_scaling_factor)
        d4 = x.detach().clone()
        x = self.softmax(x)

        if self.dense_out:
            return x, (d1, d2, d3, d4)
        if self.quant_out:
            return x, (q0, q1, q2, q3)
        return x


def jettagger_model(**kwargs):
    return MultiLayerPerceptron()

def q_jettagger_model(model=None, **kwargs):
    if model == None:
        model = MultiLayerPerceptron()
    return Q_JetTagger(model, **kwargs)
