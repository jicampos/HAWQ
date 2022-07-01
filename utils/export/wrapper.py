import copy
import logging
import warnings

import torch
import torch.nn as nn
from torch._C import ListType, OptionalType

import onnx 
import qonnx
import onnxoptimizer

from .function import *
from ..quantization_utils.quant_modules import QuantAct, QuantDropout, QuantLinear, QuantBnConv2d
from ..quantization_utils.quant_modules import QuantMaxPool2d, QuantAveragePool2d, QuantConv2d

EXPORT_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantAveragePool2d,
    QuantBnConv2d
)

SUPPORTED_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantMaxPool2d, 
    QuantAveragePool2d, 
    QuantDropout,
    QuantBnConv2d
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


# ------------------------------------------------------------
class ExportONNXQuantAct(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

        self.layer = layer
        self.export_mode = False

        self.is_binary = layer.activation_bit == 1
        self.scale = layer.act_scaling_factor.clone().detach().requires_grad_(False)

        if layer.full_precision_flag:
            self.bit_width = 32
        else:
            self.bit_width = layer.activation_bit

        self.args = (
            torch.tensor(layer.act_scaling_factor.clone().detach().item(), dtype=torch.float32),  # scale
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
            x = QuantFunc.apply(x, *self.args)
            model_info['quant_out_export_mode'][self.layer] = x
            return (x, model_info['act_scaling_factor'][self.layer])
        else:
            with torch.no_grad():
                x, act_scaling_factor = self.layer(x, pre_act_scaling_factor, pre_weight_scaling_factor,
                                                         identity, identity_scaling_factor, identity_weight_scaling_factor)
            model_info['act_scaling_factor'][self.layer] = act_scaling_factor
            model_info['quant_out'][self.layer] = x
            return (x, act_scaling_factor)


# ------------------------------------------------------------
class ExportONNXQuantLinear(nn.Module):
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

            # x = x / prev_act_scaling_factor.view(1,-1)[0].clone().detach().requires_grad_(False)
            x = torch.matmul(x, weights)

            if self.has_bias:
                QuantFunc = get_quant_func(self.layer.bias_bit)
                bias =  QuantFunc.apply(self.fc.bias.data, *self.bias_args, *self.kwargs)
                x = torch.add(x, bias)
            # x = torch.round(x)
            bias_scaling_factor = self.scale * prev_act_scaling_factor.clone().detach().requires_grad_(False)
            x = x * bias_scaling_factor
            model_info['dense_out_export_mode'][self.layer] = x.clone().detach().requires_grad_(False)
            return x
        else:
            with torch.no_grad():
                x = self.layer(x, prev_act_scaling_factor)
            model_info['dense_out'][self.layer] = x.clone().detach().requires_grad_(False)
            return x


# ------------------------------------------------------------
class ExportONNXQuantConv2d(nn.Module):
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
class ExportONNXQuantAveragePool2d(nn.Module):
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
class ExportONNXQuantBnConv2d(nn.Module):
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
            return (self.bn(x), conv_scaling_factor)
        else:
            x, convbn_scaling_factor = self.layer(x, pre_act_scaling_factor)
            model_info['convbn_scaling_factor'][self.layer] = convbn_scaling_factor
            return self.export_quant_conv(x, convbn_scaling_factor)


SET_EXPORT_MODE = (ExportONNXQuantAct, ExportONNXQuantLinear, ExportONNXQuantConv2d, ExportONNXQuantAveragePool2d, ExportONNXQuantBnConv2d)


export_layers_dict = {
    'QuantAct': ExportONNXQuantAct,
    'QuantLinear': ExportONNXQuantLinear,
    'QuantConv2d': ExportONNXQuantConv2d,
    'QuantAveragePool2d': ExportONNXQuantAveragePool2d,
    'QuantBnConv2d': QuantBnConv2d
}

# ------------------------------------------------------------
class ExportManager(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        assert model is not None, 'Model cannot be None'
        self.model_info = {}
        self.model_info['transformed'] = {}

        self.export_model = copy.deepcopy(model)
        self.replace_layers()

    def forward(self, x):
        self.set_export_mode('enable')
        x = self.export_model(x)
        self.set_export_mode('disable')
        return x

    # def replace_layers(self):
    #     for param in self.export_model.parameters():
    #         param.requires_grad_(False)

    #     for name in dir(self.export_model):
    #         layer = getattr(self.export_model, name)
    #         onnx_export_layer = None
    #         if isinstance(layer, nn.Module) and layer.__class__.__name__ in EXPORT_LAYERS:
    #             onnx_export_layer = export_layers_dict[layer.__class__.__name__]()
    #             print(layer.__class__.__name__)
    #             setattr(self.export_model, name, onnx_export_layer)
    #         # track changes 
    #         if onnx_export_layer is not None:
    #             self.model_info['transformed'][layer] = onnx_export_layer

    def replace_layers(self):
        for param in self.export_model.parameters():
            param.requires_grad_(False)

        for name in dir(self.export_model):
            layer = getattr(self.export_model, name)
            onnx_export_layer = None
            if isinstance(layer, QuantAct):
                onnx_export_layer = ExportONNXQuantAct(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantLinear):
                onnx_export_layer = ExportONNXQuantLinear(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantConv2d):
                onnx_export_layer = ExportONNXQuantConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantBnConv2d):
                onnx_export_layer = ExportONNXQuantBnConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantAveragePool2d):
                onnx_export_layer = ExportONNXQuantAveragePool2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, torch.nn.Sequential):
                # no nn.ModuleList?
                replace_layers(layer)
            # track changes 
            if onnx_export_layer is not None:
                self.model_info['transformed'][layer] = onnx_export_layer

    @staticmethod
    def enable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = True

    @staticmethod
    def disable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = False 

    def set_export_mode(self, export_mode):
        if export_mode == 'enable':
            self.export_model.apply(self.enable_export)
        else:
            self.export_model.apply(self.disable_export)
    
    def optimize_onnx_model(self, model_path):
        onnx_model = onnxoptimizer.optimize(onnx.load_model(model_path), passes=['extract_constant_to_initializer'])
        from qonnx.util.cleanup import cleanup
        cleanup(onnx_model, out_file=model_path)

    def export(self, input_tensor, filename=None, save=True):
        assert input_tensor is not None, 'Model input cannot be None'

        if filename is None:
            from datetime import datetime
            now = datetime.now() # current date and time
            date_time = now.strftime('%m%d%Y_%H%M%S')
            filename = f'results/{model._get_name()}_{date_time}.onnx'

        register_custom_ops()

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # first pass (collect scaling factors for onnx nodes) 
                self.set_export_mode('disable')
                x = self.export_model(input_tensor)
                # export with collected values 
                self.set_export_mode('enable')
                if save:
                    print('Exporting model...')
                    torch.onnx.export(
                                    model=self.export_model, 
                                    args=input_tensor, 
                                    f=filename, 
                                    opset_version=11,
                                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                                    custom_opsets={DOMAIN_STRING: 1}
                    )
                    self.optimize_onnx_model(filename)
