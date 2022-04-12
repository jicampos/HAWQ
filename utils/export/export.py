"""
Exporting the JetTagger model in FINN/ONNX format.

TODO:
    Confirm attributes of QuantAct
    Create deep copy of model before exporting 
    Start creating high level api for export 
    Organize code into files 
    Increase exception hanlding and verify input 
"""

import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
from torch.onnx import register_custom_op_symbolic

from utils import q_jettagger_model, q_resnet18
from utils.export.utils import set_export_mode
from utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantBnConv2d
from utils.quantization_utils.quant_modules import QuantMaxPool2d, QuantAveragePool2d, QuantConv2d


SUPPORTED_LAYERS = (QuantAct, 
                    QuantLinear)

UNSUPPORTED_LAYERS = (QuantBnConv2d, 
                      QuantMaxPool2d, 
                      QuantAveragePool2d, 
                      QuantConv2d)

# https://github.com/Xilinx/finn-base/tree/dev/src/finn/custom_op/general
SUPPORTED_QONNX = ['Quant']
UNSUPPORTED_QONNX = ['BipolarQuant', 'Trunc']


# ------------------------------------------------------------
class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op("Quant", 
                x, scale, zero_point, bit_width,  # must be tensors
                rounding_mode_s=rounding_mode,
                signed_i=int(signed),
                narrow_i=int(narrow_range))

register_custom_op_symbolic("::QuantAct", QuantFunc.symbolic, 1)

class BinaryQuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        return x

    @staticmethod
    def symbolic(g, x, scale):
        return g.op("BipolarQuant", x, scale)

register_custom_op_symbolic("::BipolarQuantAct", BinaryQuantFunc.symbolic, 1)


# ------------------------------------------------------------
class ExportQuantAct(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        # save a reference to QuantAct?
        self.export_mode = True

        self.scale = torch.tensor(layer.act_scaling_factor)
        self.zero_point = torch.tensor(0)
        if layer.full_precision_flag:
            self.bit_width = torch.tensor(32)
        else:
            self.bit_width = torch.tensor(layer.activation_bit)
        
        self.narrow_range = 0
        self.rounding_mode = 'ROUND'
        if layer.quant_mode == 'symmetric':
            self.signed = 1 
        else:
            self.signed = 0    
    
    def forward(self, x, act_scaling_factor=None, weight_scaling_factor=None):
        if self.export_mode:
            if self.bit_width == 1:
                return BinaryQuantFunc.apply(x, self.scale), act_scaling_factor 
            return QuantFunc.apply(x, self.scale, self.zero_point, self.bit_width,
                        self.signed, self.narrow_range, self.rounding_mode), act_scaling_factor
        else:
            return x


# ------------------------------------------------------------
class ExportQuantLinear(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        
        has_bias = hasattr(layer, 'bias')
        self.fc = torch.nn.Linear(in_features, out_features, has_bias)
        
        if has_bias:
            if layer.fix_flag:
                self.fc.weight.data = layer.weight
                self.fc.bias.data = layer.bias
            else:
                self.fc.weight.data = layer.weight_integer
                self.fc.bias.data = layer.bias_integer
        else:
            if layer.fix_flag:
                self.fc.weight.data = layer.weight
            else:
                self.fc.weight.data = layer.weight_integer

    def forward(self, x, act_scaling_factor=None, weight_scaling_factor=None):
        return self.fc(x), act_scaling_factor


# ------------------------------------------------------------
def replace_all(model):
    # model = model.features
    for param in model.parameters():
        param.requires_grad_(False)

    for name in dir(model):
        layer = getattr(model, name)
        if isinstance(layer, QuantAct):
            setattr(model, name, ExportQuantAct(layer))
        elif isinstance(layer, QuantLinear):
            setattr(model, name, ExportQuantLinear(layer))
        elif isinstance(layer, torch.nn.Sequential):
            replace_all(layer)
        elif isinstance(layer, UNSUPPORTED_LAYERS):
            raise Exception(f'Unsupported layer type found {layer._get_name()}')

def replace_with_nn(model):
    model = model.features
    for param in model.parameters():
        param.requires_grad_(False)
    
    from utils.quantization_utils.quant_modules import QuantAct, QuantLinear
    for name, layer in enumerate(model):
        if isinstance(layer, torch.nn.Sequential):
            replace_with_nn(layer)
        if isinstance(layer, QuantAct):
            print('quantact')
            model[name] = ExportQuantAct()
        if isinstance(layer, QuantLinear):
            print('quantlinear')
            model[name] = ExportQuantLinear(layer)

def replace_nn_apply(model):
    for param in model.parameters():
        param.requires_grad_(False)
    
    from utils.quantization_utils.quant_modules import QuantAct, QuantLinear
    if isinstance(model, QuantAct):
        print('quantact')
    if isinstance(model, QuantLinear):
        print('quantlinear')


# ------------------------------------------------------------
if __name__ == '__main__':
    x = torch.randn([1, 16])
    model = q_jettagger_model()

    onnx_test = False
    if onnx_test:
        out = model(x) 
        print(out) 
        set_export_mode(model, True) 
        torch.onnx.export(model, x, 'hawq_test.onnx') 
        bp = 0

    model(x)

    torch.onnx.export(model, x, 'hawq_replaced_layers_v6.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK) 
    # import torchvision.models as models
    # resnet18 = models.resnet18()
    # print(replace_all(q_resnet18(resnet18))
