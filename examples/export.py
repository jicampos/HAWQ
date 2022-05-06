"""
Exporting HAWQ models to the ONNX format.

TODO:
    High Level API Call (ONNX -> HAWQ)  
"""

import argparse
import warnings
import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
import torch.autograd
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

model_info = {}
model_info['transformed'] = {}
model_info['layer_io'] = {}

# https://github.com/Xilinx/finn-base/tree/dev/src/finn/custom_op/general
SUPPORTED_QONNX = ['Quant', 'BipolarQuant']
UNSUPPORTED_QONNX = ['Trunc']

DOMAIN_STRING = ''

# ------------------------------------------------------------
class HawqQuantFunc(torch.autograd.Function):
    name = 'Quant'

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Quant', 
                x, scale, zero_point, bit_width,  
                signed_i=int(signed),
                narrow_i=int(narrow_range),
                rounding_mode_s=rounding_mode
        )
        # ret.setType(OptionalType.kind(scale))
        # OptionalType.kind(scale)


class HawqBinaryQuantFunc(torch.autograd.Function):
    name = 'BipolarQuant'

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::BipolarQuant', x, scale)


class HawqTruncFunc(torch.autograd.Function):
    name = 'Trunc'

    @staticmethod
    def forward(ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return torch.trunc(x)

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Trunc',
                    x, scale, zero_point, input_bit_width, output_bit_width,
                    rounding_mode_s=rounding_mode)


# subtituted in for autograd.Function in quant_utils
class HawqRoundFunc(torch.autograd.Function):
    name = 'Round'

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def symbolic(g, x):
        return g.op('Round', x)


HAWQ_FUNCS = [
            HawqBinaryQuantFunc, 
            HawqQuantFunc, 
            HawqTruncFunc, 
            HawqRoundFunc
]

def get_quant_func(bit_width):
    if bit_width == 1:
        return HawqBinaryQuantFunc
    return HawqQuantFunc


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
            # torch.tensor([1], dtype=torch.float32),  # scale  
            torch.tensor(0, dtype=torch.float32),  # zero point 
            torch.tensor(self.bit_width, dtype=torch.float32),   # bit width 
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
                return HawqBinaryQuantFunc.apply(x, *self.args), pre_act_scaling_factor 
            return HawqQuantFunc.apply(x, *self.args), model_info['layer_io'][self.layer]
        else:
            x, act_scaling_factor = self.layer(x, pre_act_scaling_factor, pre_weight_scaling_factor, identity, identity_scaling_factor, identity_weight_scaling_factor)
            model_info['layer_io'][self.layer] = act_scaling_factor
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

        self.weight_args = (
            torch.tensor(1, dtype=torch.float32),                # scale  
            torch.tensor(0, dtype=torch.float32),                # zero point 
            torch.tensor(layer.weight_bit, dtype=torch.float32)  # bit width 
        )

        if self.has_bias:
            self.bias_args = (
                torch.tensor(1, dtype=torch.float32),              # scale  
                torch.tensor(0, dtype=torch.float32),              # zero point 
                torch.tensor(layer.bias_bit, dtype=torch.float32)  # bit width 
            )
        
        self.kwargs = (
            int(1 if layer.quant_mode == 'symmetric' else 0),  # signed 
            int(0),                                            # narrow range
            'ROUND'                                            # rounding mode 
        )

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]
        
        if self.export_mode:
            QuantFunc = get_quant_func(self.layer.weight_bit)
            weights = QuantFunc.apply(self.fc.weight.data, *self.weight_args, *self.kwargs)
            x = torch.matmul(x, weights)
            
            if self.has_bias:
                QuantFunc = get_quant_func(self.layer.bias_bit)
                bias =  QuantFunc.apply(self.fc.bias.data, *self.bias_args, *self.kwargs)
                x = torch.add(x, bias)
            return x, model_info['layer_io'][self.layer]
        else:
            x, act_scaling_factor = self.layer(x, prev_act_scaling_factor)
            model_info['layer_io'][self.layer] = act_scaling_factor
            return (x, act_scaling_factor)


# ------------------------------------------------------------
class ExportONNXQuantConv2d(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.conv = layer.conv
        self.weight_integer = model.conv1.weight_integer

    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        if x.ndim == 3:
            x = x[None] # add dimension for batch size, converts to 4D tensor
        
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
            torch.tensor(1),   # scale 
            torch.tensor(0),   # zero point 
            torch.tensor(32),  # input bit width
            torch.tensor(32),  # output bit width
            'ROUND'            # rounding mode 
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
        x_int = HawqRoundFunc.apply(x_int) 
        x_int = self.pool(x_int)

        eps = 0.01
        x_int = HawqTruncFunc.apply(x_int+eps, *self.trunc_args)

        return (x_int * correct_scaling_factor, correct_scaling_factor)

# ------------------helper functions------------------ 
SET_EXPORT_MODE = (ExportONNXQuantAct, ExportONNXQuantLinear)

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
    
EXPORT_LOOKUP = {
    QuantAct: ExportONNXQuantAct,
    QuantLinear: ExportONNXQuantLinear,
    QuantConv2d: ExportONNXQuantConv2d,
    QuantAveragePool2d: ExportONNXQuantAveragePool2d
}

EXPORT_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantAveragePool2d
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
        elif isinstance(layer, QuantAveragePool2d):
            onnx_export_layer = ExportONNXQuantAveragePool2d(layer)
            setattr(model, name, onnx_export_layer)
        elif isinstance(layer, torch.nn.Sequential):
            replace_layers(layer)
        elif isinstance(layer, UNSUPPORTED_LAYERS):
            raise RuntimeError(f'Unsupported layer type found {layer._get_name()}')
        
        # track changes 
        if onnx_export_layer is not None:
            model_info['transformed'][layer] = onnx_export_layer

def register_custom_ops():
    for func in HAWQ_FUNCS:
        register_custom_op_symbolic(f'{DOMAIN_STRING}::{func.name}', func.symbolic, 1)

def rename_node_vars(onnx_model):
    # rename node names 
    name_counter = 0
    
    for node in onnx_model.graph.node:
        if node.name == '':
            node.name = f'node_{name_counter}'
            name_counter += 1
    
    # change input name 
    output =[node.name for node in onnx_model.graph.output]

    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))

    print('Inputs: ', net_feed_input)
    print('Outputs: ', output)

    input_nodes = [node for node in onnx_model.graph.input if node.name in net_feed_input]
    in_node = input_nodes[0]
    print(in_node)

    default_name = in_node.name 
    in_node.name = 'global_in'

    print(in_node)

    # change all references to default input name 
    for node in model.graph.node: 
        if default_name in node.input:
            for i, input in enumerate(node.input):
                if input == default_name:
                    node.input[i] = 'global_in'

def optimize_onnx_model(model_path):
    model_opt_path = f'{model_path.split(".")[0]}_optimized.onnx'

    # change constant operators to initializer 
    onnx_model = onnxoptimizer.optimize(onnx.load_model(model_path), passes=['extract_constant_to_initializer'])
    # rename each node, input and input references  
    rename_node_vars(onnx_model)

    onnx.save_model(onnx_model, model_opt_path)

def export_to_qonnx(model, input, filename=None, save=True):
    if model is None:
        raise ValueError('Model cannot be None')
    if input is None:
        raise ValueError('Input cannot be None')

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
            # first pass (collect scaling factors for onnx nodes) 
            set_export_mode(export_model, 'disable')
            x = export_model(input)
            # export with collected values 
            set_export_mode(export_model, 'enable')
            if save:
                print('Exporting model...')
                torch.onnx.export(
                    model=export_model, 
                    args=input, 
                    f=filename, 
                    keep_initializers_as_inputs=True,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                    custom_opsets={DOMAIN_STRING: 1})
                
                optimize_onnx_model(filename)
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
        en_original_step_size=[48,48]
        pool_list=[2,2,2]

        de_original_step_size = [4,4]
        up_list = [2,2,3]
        embedding_size=6
        conv_size =128
        from q_hawq_tem import conv_block, hawq_conv_block, HawqEncoder, Encoder
        # model = conv_block(conv_size, en_original_step_size)
        # model = hawq_conv_block(model, conv_size, en_original_step_size)
        encoder = Encoder(original_step_size=en_original_step_size,
                  pool_list=pool_list,
                  embedding_size=embedding_size,
                  conv_size=conv_size)
        print(encoder)
        model = HawqEncoder(model=encoder,original_step_size=en_original_step_size,
                  pool_list=pool_list,
                  embedding_size=embedding_size,
                  conv_size=conv_size)
        x = torch.randn([1, 1, 48, 48])
        print(encoder)
    
    print('Original layers:')
    print('-----------------------------------------------------------------------------')
    print(model)
    print('-----------------------------------------------------------------------------')

    # from save_checkpoint import load_checkpoint
    # load_checkpoint(model, 'checkpoint.pth.tar')
    export_model = export_to_qonnx(model, x, save=True)

    print('New layers:')
    print('-----------------------------------------------------------------------------')
    print(export_model)
    print('-----------------------------------------------------------------------------')

    print('\t{:12}     {}'.format('Original', 'New'))
    print('-----------------------------------------------------------------------------')
    for org, export in model_info['transformed'].items():
        print('\t{:12} --> {}'.format(org._get_name(), export._get_name()))
    print('-----------------------------------------------------------------------------\n')

    # print('Layer IO:')
    print('\t{:12}  {}'.format('Layer', 'Scaling Factor'))
    print('-----------------------------------------------------------------------------')
    for layer, out in model_info['layer_io'].items():
        print(f'\t{layer._get_name():12}: {out}')
    print('-----------------------------------------------------------------------------')
