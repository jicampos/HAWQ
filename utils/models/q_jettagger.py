"""
    3 Hidden Layer Jet Tagging Model
"""

import torch.nn as nn
from ..quantization_utils.quant_modules import QuantAct, QuantLinear


class three_layer_model(nn.Module):
    def __init__(self):
        """Multi-Layer Perceptron with <16,64,32,32,5> behavior"""
        super(three_layer_model, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        softmax_out = self.softmax(self.fc4(x))
        return softmax_out


class Q_JetTagger(nn.Module):
    """
    Quantized JetTagger model.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point JetTagger.
    weight_precision: int, default 8
        Bitwidth for quantized weights.
    bias_precision: int, default 32
        Bitwidth for quantized bias.
    act_precision: int, default 16
        Bitwidth for quantized activations.
    """
    def __init__(self, 
                 model, 
                 weight_precision=8,
                 bias_precision=32,
                 act_precision=16):
        super(Q_JetTagger, self).__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        self.act_precision = act_precision
        
        self.quant_input = QuantAct(act_precision)

        self.quant_act1 = QuantAct(act_precision)
        self.quant_act2 = QuantAct(act_precision)
        self.quant_act3 = QuantAct(act_precision)
        self.quant_act4 = QuantAct(act_precision)

        self.features = nn.Sequential()

        fc1_data = getattr(model, 'fc1')
        self.features.add_module("fc1", QuantLinear(weight_precision,bias_bit=bias_precision))
        self.features.add_module("act1", nn.ReLU6())
        self.features.fc1.set_param(fc1_data)

        fc2_data = getattr(model, 'fc2')
        self.features.add_module("fc2", QuantLinear(weight_precision,bias_bit=bias_precision))
        self.features.add_module("act2", nn.ReLU6())
        self.features.fc2.set_param(fc2_data)

        fc3_data = getattr(model, 'fc3')
        self.features.add_module("fc3", QuantLinear(weight_precision,bias_bit=bias_precision))
        self.features.add_module("act3", nn.ReLU6())
        self.features.fc3.set_param(fc3_data)

        fc4_data = getattr(model, 'fc4')
        self.features.add_module("fc4", QuantLinear(weight_precision,bias_bit=bias_precision))
        self.features.add_module("softmax", nn.Softmax(0))
        self.features.fc4.set_param(fc4_data)

    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)
        # the FC1 block, 16 -> 64
        x, weight_scaling_factor = self.features.fc1(x, act_scaling_factor)
        x = self.features.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)
        # the FC2 block, 64 -> 32
        x, weight_scaling_factor = self.features.fc2(x, act_scaling_factor)
        x = self.features.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)
        # the FC3 block, 32 -> 32
        x, weight_scaling_factor = self.features.fc3(x, act_scaling_factor)
        x = self.features.act3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, weight_scaling_factor)
        # the FC4/Output block, 32 -> 5
        x, weight_scaling_factor = self.features.fc4(x, act_scaling_factor)
        x = self.features.softmax(x)
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, weight_scaling_factor)
        x = x.view(x.size(0), -1)
        return x


def q_jettagger_model(model=None):
    if model == None:
        model = three_layer_model()
    return Q_JetTagger(model)



