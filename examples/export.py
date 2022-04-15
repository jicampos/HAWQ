"""
Exporting HAWQ models to the ONNX format.

TODO:
    High Level API Call (PyTorch -> HAWQ)
"""

import warnings
import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
from torch.onnx import register_custom_op_symbolic
from torch._C import ListType, OptionalType

from utils import q_jettagger_model, q_resnet18, q_mnist
from utils.export.utils import set_export_mode
from utils.quantization_utils.quant_modules import QuantAct, QuantDropout, QuantLinear, QuantBnConv2d
from utils.quantization_utils.quant_modules import QuantMaxPool2d, QuantAveragePool2d, QuantConv2d


SUPPORTED_LAYERS = (QuantAct, 
                    QuantLinear,
                    QuantConv2d, 
                    QuantMaxPool2d, 
                    QuantAveragePool2d, 
                    QuantDropout)

UNSUPPORTED_LAYERS = (QuantBnConv2d)

model_stats = {}

# https://github.com/Xilinx/finn-base/tree/dev/src/finn/custom_op/general
SUPPORTED_QONNX = ['Quant', 'BipolarQuant']
UNSUPPORTED_QONNX = ['Trunc']

DOMAIN_STRING = 'hawq_onnx'

# ------------------------------------------------------------
class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        ret = g.op(f'{DOMAIN_STRING}::Quant', 
                x, scale, zero_point, bit_width,  
                signed_i=int(signed),
                narrow_i=int(narrow_range),
                rounding_mode_s=rounding_mode
        )
        ret.setType(OptionalType.ofTensor())
        return ret


class BinaryQuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        return x

    @staticmethod
    def symbolic(g, x, scale):
        return g.op(f'{DOMAIN_STRING}::BipolarQuant', x, scale)


class TruncFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Trunc',
                    x, scale, zero_point, input_bit_width, output_bit_width,
                    rounding_mode_s=rounding_mode)

# ------------------------------------------------------------
class ExportONNXQuantAct(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        
        self.is_binary = layer.activation_bit == 1
        self.scale = layer.act_scaling_factor.clone().detach().requires_grad_(False)

        if layer.full_precision_flag:
            self.bit_width = torch.tensor(32)
        else:
            self.bit_width = torch.tensor(layer.activation_bit)

        self.args = (
            torch.tensor(1),  # scale  
            torch.tensor(0),  # zero point 
            torch.tensor(self.bit_width),   # bit width 
            int(1 if layer.quant_mode == 'symmetric' else 0),  # signed 
            int(0),  # narrow range
            'ROUND'  # rounding mode 
        )
    
    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None,
                identity_scaling_factor=None, identity_weight_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        
        if self.is_binary:
            return BinaryQuantFunc.apply(x, self.scale), pre_act_scaling_factor 
        return QuantFunc.apply(x, *self.args), pre_act_scaling_factor


# ------------------------------------------------------------
class ExportONNXQuantLinear(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

        self.has_bias = hasattr(layer, 'bias')
        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        self.fc = torch.nn.Linear(in_features, out_features, self.has_bias) 
        
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x9216 and 1x9216)
        if self.has_bias:
            self.fc.weight.data = torch.transpose(layer.weight_integer, 0, 1)
            self.fc.bias.data = layer.bias_integer
        else:
            self.fc.weight.data = torch.transpose(layer.weight_integer, 0, 1)
        
        self.weight_args = (
            torch.tensor(1),  # scale  
            torch.tensor(0),  # zero point 
            torch.tensor(layer.weight_bit)  # bit width 
        )

        if self.has_bias:
            self.bias_args = (
                torch.tensor(1),  # scale  
                torch.tensor(0),  # zero point 
                torch.tensor(layer.bias_bit)    # bit width 
            )
        
        self.kwargs = (
            int(1 if layer.quant_mode == 'symmetric' else 0),  # signed 
            int(0),  # narrow range
            'ROUND'  # rounding mode 
        )

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]
        
        weights = QuantFunc.apply(self.fc.weight.data, *self.weight_args, *self.kwargs)
        x = torch.matmul(x, weights)

        if self.has_bias:
            biases =  QuantFunc.apply(self.fc.bias.data, *self.bias_args, *self.kwargs)
            x = torch.add(x, biases)
        return x, prev_act_scaling_factor


# ------------------------------------------------------------
class ExportONNXQuantConv2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.conv = layer.conv

    def forward(self, x, act_scaling_factor=None):
        return self.conv(x), act_scaling_factor


# ------------------------------------------------------------
class ExportONNXQuantBnConv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        
    def forward(self, x, pre_act_scaling_factor=None):
        ...
    
    
class ExportQuantAveragePool2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, x_scaling_factor=None):
        ...

# ------------------------------------------------------------
def replace_layers(model):
    for param in model.parameters():
        param.requires_grad_(False)

    for name in dir(model):
        layer = getattr(model, name)
        if isinstance(layer, QuantAct):
            setattr(model, name, ExportONNXQuantAct(layer))
        elif isinstance(layer, QuantLinear):
            setattr(model, name, ExportONNXQuantLinear(layer))
        elif isinstance(layer, QuantConv2d):
            setattr(model, name, ExportONNXQuantConv2d(layer))
        elif isinstance(layer, torch.nn.Sequential):
            replace_layers(layer)
        elif isinstance(layer, UNSUPPORTED_LAYERS):
            raise RuntimeError(f'Unsupported layer type found {layer._get_name()}')


# ------------------------------------------------------------
def register_custom_ops():
    register_custom_op_symbolic(f'{DOMAIN_STRING}::Quant', QuantFunc.symbolic, 1)
    register_custom_op_symbolic(f'{DOMAIN_STRING}::BipolarQuant', BinaryQuantFunc.symbolic, 1)

def export_to_qonnx(model, input, filename=None):
    if model is None:
        raise RuntimeError('Model cannot be None')
    if input is None:
        raise RuntimeError('Input cannot be None')

    if filename is None:
        from datetime import datetime
        now = datetime.now() # current date and time
        date_time = now.strftime('%m%d%Y%H%M%S')
        filename = f'{model._get_name()}_{date_time}.onnx'

    register_custom_ops()
    import copy 
    export_model = copy.deepcopy(model)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            replace_layers(export_model)
            torch.onnx.export(export_model, input, filename)
        
    return export_model

# ------------------------------------------------------------
if __name__ == '__main__':
    
    is_jet = False
    if is_jet:
        x = torch.randn([1, 16])
        model = q_jettagger_model()
    else:
        x = torch.randn([1, 1, 28, 28])
        model = q_mnist()

    export_model = export_to_qonnx(model, x)

    # import torchvision.models as models
    # resnet18 = models.resnet18()
    # print(replace_all(q_resnet18(resnet18))
