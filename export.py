"""
    Exporting HAWQ models to the ONNX format.
"""
import os
import sys
import logging
import argparse
import warnings

import onnx
import onnxoptimizer
import qonnx

import torch
import torch.autograd
from torch.onnx import register_custom_op_symbolic
from torch._C import ListType, OptionalType

from utils import q_jettagger_model, q_resnet18, q_mnist
from utils.quantization_utils.quant_modules import QuantAct, QuantDropout, QuantLinear, QuantBnConv2d
from utils.quantization_utils.quant_modules import QuantMaxPool2d, QuantAveragePool2d, QuantConv2d

from args import *

SUPPORTED_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantMaxPool2d, 
    QuantAveragePool2d, 
    QuantDropout,
    QuantBnConv2d
)

UNSUPPORTED_LAYERS = (

)

model_info = {}
model_info['dense_out'] = dict()
model_info['dense_out_export_mode'] = dict()
model_info['quant_out'] = dict()
model_info['quant_out_export_mode'] = dict()
model_info['transformed'] = dict()
model_info['act_scaling_factor'] = dict()
model_info['conv_scaling_factor'] = dict()
model_info['convbn_scaling_factor'] = dict()

SUPPORTED_QONNX_OPS = ['Quant', 'Trunc']
UNSUPPORTED_QONNX_OPS = ['BipolarQuant']

DOMAIN_STRING = 'hawq2qonnx'

# ------------------------------------------------------------
class QuantFunc(torch.autograd.Function):
    name = 'Quant'

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        min_ = -1*(2**bit_width)
        max_ = (2**bit_width)-1
        return torch.clamp(torch.round((x/scale)+zero_point), min_, max_)

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Quant', 
                x, scale, zero_point, bit_width,  
                signed_i=int(signed),
                narrow_i=int(narrow_range),
                rounding_mode_s=rounding_mode
        )

class BinaryQuantFunc(torch.autograd.Function):
    name = 'BipolarQuant'

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::BipolarQuant', x, scale)

class TruncFunc(torch.autograd.Function):
    name = 'Trunc'

    @staticmethod
    def forward(ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return torch.trunc(x)

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Trunc',
                    x, scale, zero_point, input_bit_width, output_bit_width,
                    rounding_mode_s=rounding_mode)

class ConvFunc(torch.autograd.Function):
    name = 'Conv'

    @staticmethod
    def forward(ctx, x, quant_input, layer, dilations, group, kernel_shape, pads, strides):
        return layer.conv(x)

    @staticmethod
    def symbolic(g, x, quant_input, layer, dilations, group, kernel_shape, pads, strides):
        return g.op(f'{DOMAIN_STRING}::Conv', x, quant_input, 
                        dilations_i=dilations, group_i=group, kernel_shape_i=kernel_shape, pads_i=pads, strides_i=strides)

class RoundFunc(torch.autograd.Function):
    name = 'Round'

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def symbolic(g, x):
        return g.op(f'{DOMAIN_STRING}::Round', x)


HAWQ_FUNCS = [
    BinaryQuantFunc, 
    QuantFunc, 
    TruncFunc,
    RoundFunc,
    ConvFunc
]

def get_quant_func(bit_width):
    if bit_width == 1:
        return BinaryQuantFunc
    return QuantFunc


# ------------------------------------------------------------
class ExportONNXLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def update_args(self):
        ...

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
            torch.tensor(layer.act_scaling_factor.detach().item(), dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),                                         # zero point
            torch.tensor(self.bit_width, dtype=torch.float32),                            # bit width
            int(1 if layer.quant_mode == 'symmetric' else 0),                             # signed
            int(0),                                                                       # narrow range
            'ROUND'                                                                       # rounding mode
        )

    def __repr__(self):
        s = f'{self.__class__.__name__}(scale={self.scale.detach().item()}, zero_point=0, bitwidth={self.layer.activation_bit},' \
                                    + f' full_precision_flag={self.layer.full_precision_flag}, quant_mode={self.layer.quant_mode})'
        return s

    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None, identity_scaling_factor=None, identity_weight_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            QuantFunc = get_quant_func(self.bit_width)
            x = QuantFunc.apply(x, *self.args)*self.scale
            model_info['quant_out_export_mode'][self.layer] = x
            return (x, model_info['act_scaling_factor'][self.layer])
        else:
            x, act_scaling_factor = self.layer(x, pre_act_scaling_factor, pre_weight_scaling_factor,
                                                         identity, identity_scaling_factor, identity_weight_scaling_factor)
            model_info['act_scaling_factor'][self.layer] = act_scaling_factor
            model_info['quant_out'][self.layer] = x
            return (x, act_scaling_factor)


# ------------------------------------------------------------
class ExportONNXQuantLinear(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

        self.layer = layer
        self.export_mode = False

        self.has_bias = hasattr(layer, 'bias')
        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        self.fc = torch.nn.Linear(in_features, out_features, self.has_bias) 

        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x9216 and 1x9216)
        self.fc.weight.data = torch.transpose(layer.weight_integer, 0, 1)
        if self.has_bias:
            self.fc.bias.data = layer.bias_integer

        self.scale = self.layer.fc_scaling_factor.clone().requires_grad_(False)
        self.weight_args = (
            torch.tensor(1, dtype=torch.float32),                    # scale  
            torch.tensor(0, dtype=torch.float32),                    # zero point 
            torch.tensor(self.layer.weight_bit, dtype=torch.float32) # bit width 
        )

        if self.has_bias:
            self.bias_args = (
                torch.tensor(1, dtype=torch.float32),              # scale  
                torch.tensor(0, dtype=torch.float32),              # zero point 
                torch.tensor(self.layer.bias_bit, dtype=torch.float32)  # bit width 
            )

        self.kwargs = (
            int(1 if self.layer.quant_mode == 'symmetric' else 0),  # signed 
            int(0),                                                 # narrow range
            'ROUND'                                                 # rounding mode 
        )
    
    def __repr__(self):
        s = f'{self.__class__.__name__}(weight_bit={self.layer.weight_bit},' \
                                    + f' bias_bit={self.layer.bias_bit}, quantize={self.layer.quant_mode})'
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            QuantFunc = get_quant_func(self.layer.weight_bit)
            weights = QuantFunc.apply(self.fc.weight.data, *self.weight_args, *self.kwargs)

            x = x / prev_act_scaling_factor.view(1,-1)[0].detach()
            x = torch.matmul(x, weights)

            if self.has_bias:
                QuantFunc = get_quant_func(self.layer.bias_bit)
                bias =  QuantFunc.apply(self.fc.bias.data, *self.bias_args, *self.kwargs)
                x = torch.add(x, bias)
            x = torch.round(x)
            bias_scaling_factor = self.scale * prev_act_scaling_factor.detach()
            x = x * bias_scaling_factor
            model_info['dense_out_export_mode'][self.layer] = x
            return x
        else:
            x = self.layer(x, prev_act_scaling_factor)
            model_info['dense_out'][self.layer] = x
            return x


# ------------------------------------------------------------
class ExportONNXQuantConv2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

        self.layer = layer
        self.export_mode = False

        self.quant_args = (
            torch.tensor(self.layer.conv_scaling_factor, dtype=torch.float32),  # scale  
            torch.tensor(0, dtype=torch.float32),                               # zero point 
            torch.tensor(self.layer.weight_bit, dtype=torch.float32),           # bit width 
            int(1 if layer.quant_mode == 'symmetric' else 0),                   # signed 
            int(0),                                                             # narrow range
            'ROUND'                                                             # rounding mode 
        )

        dilation = self.layer.conv.dilation
        if type(dilation) != tuple:
            dilation = (dilation, dilation)

        kernel_size = self.layer.conv.kernel_size
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, kernel_size)

        pads = self.layer.conv.padding
        if type(pads) != tuple:
            pads = (pads, pads, pads, pads)

        strides = self.layer.conv.stride
        if type(strides) != tuple:
            strides = (strides, strides)

        self.conv_args = (
            dilation,
            self.layer.conv.groups,
            kernel_size,
            pads,
            strides
        )

    def __repr__(self):
        s = f'{self.__class__.__name__}()'
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        if x.ndim == 3:
            # add dimension for batch size, 4D tensor
            x = x[None] 

        if self.export_mode:
            QuantFunc = get_quant_func(self.layer.weight_bit)
            weights = QuantFunc.apply(self.layer.weight_integer, *self.quant_args)
            return (HawqConvFunc.apply(x, weights, self.layer, *self.conv_args), model_info['conv_scaling_factor'][self.layer])
        else:
            x, conv_scaling_factor = self.layer(x, prev_act_scaling_factor)
            model_info['conv_scaling_factor'][self.layer] = conv_scaling_factor
            print(x.shape)
            return (x, conv_scaling_factor)


# ------------------------------------------------------------
class ExportONNXQuantAveragePool2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        
        self.layer = layer
        self.export_mode = False

        self.trunc_args = (
            torch.tensor(1),   # scale 
            torch.tensor(0),   # zero point 
            torch.tensor(32),  # input bit width
            torch.tensor(32),  # output bit width
            'ROUND'            # rounding mode 
        )

    def __repr__(self):
        s = f'{self.__class__.__name__}()'
        return s

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]
        
        if x_scaling_factor is None:
            return (self.layer(x), x_scaling_factor)
        
        if self.export_mode:
            x_scaling_factor = x_scaling_factor.view(-1)
            correct_scaling_factor = x_scaling_factor

            x_int = x / correct_scaling_factor
            x_int = HawqRoundFunc.apply(x_int) 
            x_int = self.layer.final_pool(x_int)

            x_int = HawqTruncFunc.apply(x_int+0.01, *self.trunc_args)

            return (x_int * correct_scaling_factor, correct_scaling_factor)
        else:
            return (self.layer(x), x_scaling_factor)


# ------------------------------------------------------------
class ExportONNXQuantBnConv2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

        self.layer = layer
        self.export_mode = False

        quant_layer = QuantConv2d()
        quant_layer.set_param(self.layer.conv)
        self.export_quant_conv = ExportONNXQuantConv2d(quant_layer)

        self.bn = torch.nn.BatchNorm2d(
            self.layer.bn.num_features,
            self.layer.bn.eps,
            self.layer.bn.momentum
        )
        self.bn.weight = self.layer.bn.weight
        self.bn.bias = self.layer.bn.bias

        self.bn.running_mean = self.layer.bn.running_mean
        self.bn.running_var = self.layer.bn.running_var

    def __repr__(self):
        s = f'{self.__class__.__name__}()'
        return s

    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            x, conv_scaling_factor = self.export_quant_conv(x)
            print(self.bn)
            return (self.bn(x), conv_scaling_factor)
        else:
            x, convbn_scaling_factor = self.layer(x, pre_act_scaling_factor)
            model_info['convbn_scaling_factor'][self.layer] = convbn_scaling_factor
            return self.export_quant_conv(x, convbn_scaling_factor)


# ------------------helper functions------------------ 
SET_EXPORT_MODE = (ExportONNXQuantAct, ExportONNXQuantLinear, ExportONNXQuantConv2d, ExportONNXQuantAveragePool2d, ExportONNXQuantBnConv2d)

def enable_export(module):
    if isinstance(module, SET_EXPORT_MODE):
        module.export_mode = True

def disable_export(module):
    if isinstance(module, SET_EXPORT_MODE):
        module.export_mode = False 

def set_export_mode(module, export_mode):
    if export_mode == 'enable':
        module.apply(enable_export)
    else:
        module.apply(disable_export)

EXPORT_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantAveragePool2d,
    QuantBnConv2d
)

def replace_layers(model):
    # use dictionary to map HAWQ layers to their Export counter part  
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
        elif isinstance(layer, QuantBnConv2d):
            onnx_export_layer = ExportONNXQuantBnConv2d(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, QuantAveragePool2d):
            onnx_export_layer = ExportONNXQuantAveragePool2d(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, torch.nn.Sequential):
            # no nn.ModuleList?
            replace_layers(layer)
        elif isinstance(layer, UNSUPPORTED_LAYERS):
            raise RuntimeError(f'Unsupported layer type found {layer._get_name()}')
        # track changes 
        if onnx_export_layer is not None:
            model_info['transformed'][layer] = onnx_export_layer

def register_custom_ops():
    for func in HAWQ_FUNCS:
        register_custom_op_symbolic(f'{DOMAIN_STRING}::{func.name}', func.symbolic, 1)

def optimize_onnx_model(model_path):
    onnx_model = onnxoptimizer.optimize(onnx.load_model(model_path), passes=['extract_constant_to_initializer'])
    from qonnx.util.cleanup import cleanup
    cleanup(onnx_model, out_file=model_path)

def export_to_qonnx(model, input_tensor, filename=None, save=True):
    assert model is not None, 'Model cannot be None'
    assert input_tensor is not None, 'Model input cannot be None'

    if filename is None:
        from datetime import datetime
        now = datetime.now() # current date and time
        date_time = now.strftime('%m%d%Y_%H%M%S')
        filename = f'results/{model._get_name()}_{date_time}.onnx'

    register_custom_ops()
    import copy 
    export_model = copy.deepcopy(model)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            replace_layers(export_model)
            # first pass (collect scaling factors for onnx nodes) 
            set_export_mode(export_model, 'disable')
            x = export_model(input_tensor)
            # export with collected values 
            set_export_mode(export_model, 'enable')
            if save:
                print('Exporting model...')
                torch.onnx.export(
                                model=export_model, 
                                args=input_tensor, 
                                f=filename, 
                                opset_version=11,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                                custom_opsets={DOMAIN_STRING: 1}
                )
                optimize_onnx_model(filename)
    return export_model


# python export.py --arch hawq_jettagger --load uniform6/06252022_152008
# ------------------------------------------------------------
if __name__ == '__main__':

    print(f'Loading {args.arch}...\n')

    if args.arch == 'hawq_jettagger':
        filename = 'hawq2qonnx_jet.onnx'
        x = torch.randn([1, 16])
        model = q_jettagger_model(model=None, dense_out=args.dense_out, quant_out=args.quant_out,
                                        batchnorm=args.batch_norm, silu=args.silu, gelu=args.gelu)
    elif args.arch == 'hawq_mnist':
        filename = 'hawq2qonnx_conv.onnx'
        x = torch.randn([1, 1, 28, 28])
        model = q_mnist()

    print('Original layers:')
    print('-----------------------------------------------------------------------------')
    print(model)
    print('-----------------------------------------------------------------------------')

    if args.load:
        quant_scheme, date_tag = args.load.split('/')
        filename = f'fixed_hawq2qonnx_jet_{quant_scheme}_{date_tag}.onnx'
        from train_utils import load_checkpoint
        load_checkpoint(model, f'checkpoints/{args.load}/model_best.pth.tar', args)

    export_model = export_to_qonnx(model, x, filename=filename, save=True)
