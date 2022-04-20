"""
Exporting HAWQ models to the ONNX format.

TODO:
    High Level API Call (PyTorch -> HAWQ)
"""

import argparse
import warnings
import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
from torch.onnx import register_custom_op_symbolic
from torch._C import ListType, OptionalType

from utils import q_jettagger_model, q_resnet18, q_mnist
# from utils.export.utils import set_export_mode
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
model_stats['transformed'] = {}
model_stats['layer_io'] = {}

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
        return torch.trunc(x)

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Trunc',
                    x, scale, zero_point, input_bit_width, output_bit_width,
                    rounding_mode_s=rounding_mode)


# subtituted in for autograd.Function in quant_utils
class RoundFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def symbolic(g, x):
        return g.op('Round', x)


# ------------------------------------------------------------
class ExportONNXQuantAct(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        
        self.layer = layer
        self.export_mode = False

        self.is_binary = layer.activation_bit == 1
        self.scale = layer.act_scaling_factor.clone().detach().requires_grad_(False)

        if layer.full_precision_flag:
            self.bit_width = torch.tensor(32)
        else:
            self.bit_width = torch.tensor(layer.activation_bit)

        self.args = (
            torch.tensor(layer.act_scaling_factor.detach().item()),  # scale  
            torch.tensor(0),  # zero point 
            torch.tensor(self.bit_width),   # bit width 
            int(1 if layer.quant_mode == 'symmetric' else 0),  # signed 
            int(0),  # narrow range
            'ROUND'  # rounding mode 
        )
    
    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None, identity_scaling_factor=None, identity_weight_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        
        if self.export_mode:
            if self.is_binary:
                return BinaryQuantFunc.apply(x, self.scale), pre_act_scaling_factor 
            return QuantFunc.apply(x, *self.args), model_stats['layer_io'][self.layer]
        else:
            x, act_scaling_factor = self.layer(x, pre_act_scaling_factor, pre_weight_scaling_factor, identity, identity_scaling_factor, identity_weight_scaling_factor)
            model_stats['layer_io'][self.layer] = act_scaling_factor
            return (x, act_scaling_factor)


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

    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        if x.ndim == 3:
            # add dimension for batch size, converts to 4D tensor
            x = x[None]
        
        return self.conv(x), pre_act_scaling_factor


# ------------------------------------------------------------
class ExportONNXQuantBnConv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x, pre_act_scaling_factor=None):
        ...


class ExportONNXQuantAveragePool2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.pool = layer.final_pool

        self.trunc_args = (
            torch.tensor(1),  # scale 
            torch.tensor(0),  # zero point 
            torch.tensor(32),  # input bit width
            torch.tensor(32),  # output bit width
            'ROUND'  # rounding mode 
        )

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]
        
        if x_scaling_factor is None:
            return (self.pool(x), x_scaling_factor)
        
        x_scaling_factor = x_scaling_factor.view(-1)
        correct_scaling_factor = x_scaling_factor

        x_int = x / correct_scaling_factor
        # x_int = torch.round(x)  # torch.round() is not differentiable
        x_int = RoundFunc.apply(x_int) 
        x_int = self.pool(x_int)

        eps = 0.01
        # x_int = torch.trunc(x_int + eps)  # torch.trunc() is not differentiable
        x_int = TruncFunc.apply(x_int+eps, *self.trunc_args)

        return (x_int * correct_scaling_factor, correct_scaling_factor)

# ------------------helper functions------------------ 
def enable_export(module):
    if isinstance(module, ExportONNXQuantAct):
        module.export_mode = True

def disable_export(module):
    if isinstance(module, ExportONNXQuantAct):
        module.export_mode = False 

def set_export_mode(module, export_mode):
    if export_mode == 'enable':
        module.apply(enable_export)
    else:
        module.apply(disable_export)

def replace_layers(model):
    for param in model.parameters():
        param.requires_grad_(False)

    for name in dir(model):
        layer = getattr(model, name)
        onnx_export_layer = None

        if isinstance(layer, QuantAct):
            onnx_export_layer = ExportONNXQuantAct(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, QuantLinear):
            onnx_export_layer = ExportONNXQuantLinear(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, QuantConv2d):
            onnx_export_layer = ExportONNXQuantConv2d(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, QuantAveragePool2d):
            onnx_export_layer = ExportONNXQuantAveragePool2d(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, torch.nn.Sequential):
            replace_layers(layer)
        elif isinstance(layer, UNSUPPORTED_LAYERS):
            raise RuntimeError(f'Unsupported layer type found {layer._get_name()}')
        
        # track changes 
        if onnx_export_layer is not None:
            model_stats['transformed'][layer] = onnx_export_layer

def register_custom_ops():
    register_custom_op_symbolic(f'{DOMAIN_STRING}::Quant', QuantFunc.symbolic, 1)
    register_custom_op_symbolic(f'{DOMAIN_STRING}::BipolarQuant', BinaryQuantFunc.symbolic, 1)
    # register_custom_op_symbolic(f'::Round', RoundFunc.symbolic, 1)

def export_to_qonnx(model, input, filename=None):
    if model is None:
        raise RuntimeError('Model cannot be None')
    if input is None:
        raise RuntimeError('Input cannot be None')

    if filename is None:
        from datetime import datetime
        now = datetime.now() # current date and time
        date_time = now.strftime('%m%d%Y%H%M%S')
        filename = f'results/{model._get_name()}_{date_time}.onnx'

    register_custom_ops()
    import copy 
    export_model = copy.deepcopy(model)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            replace_layers(export_model)
            print('Exporting model...')
            # first pass (collect scaling factors for onnx nodes)
            set_export_mode(export_model, 'disable')
            x = export_model(input)
            # export with collected values 
            set_export_mode(export_model, 'enable')
            torch.onnx.export(export_model, input, filename, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        
    return export_model

# ------------------------------------------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Options for exporting.')
    parser.add_argument('--model', '-m', help='Predefined model to export', type=str, default='Q_JetTagger')
    args = parser.parse_args()

    print(f'Loading {args.model}...\n')

    if args.model == 'Q_JetTagger':
        x = torch.randn([1, 16])
        model = q_jettagger_model()
    elif args.model == 'Q_MNIST':
        x = torch.randn([1, 1, 28, 28])
        model = q_mnist()
    else:
        print(f'Unrecognized model {args.model}')
        print('Using Q_JetTagger instead.')
        x = torch.randn([1, 16])
        model = q_jettagger_model()
    
    print('Original layers:')
    print('-----------------------------------------------------------------------------')
    print(model)
    print('-----------------------------------------------------------------------------')

    export_model = export_to_qonnx(model, x)

    print('New layers:')
    print('-----------------------------------------------------------------------------')
    print(export_model)
    print('-----------------------------------------------------------------------------')

    print('Layer changes...')
    print('-----------------------------------------------------------------------------')
    for org, export in model_stats['transformed'].items():
        print('\t{:18} --> {}'.format(org._get_name(), export._get_name()))
    print('-----------------------------------------------------------------------------')

    print('Layer IO:')
    print('-----------------------------------------------------------------------------')
    for layer, out in model_stats['layer_io'].items():
        print(f'{layer._get_name()}: {out}')
    print('-----------------------------------------------------------------------------')
