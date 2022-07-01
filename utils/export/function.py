import torch
import torch.autograd as autograd
from torch.onnx import register_custom_op_symbolic

DOMAIN_STRING = 'hawq2qonnx'


class QuantFunc(autograd.Function):
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


class BinaryQuantFunc(autograd.Function):
    name = 'BipolarQuant'

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return x

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::BipolarQuant', x, scale)


class TruncFunc(autograd.Function):
    name = 'Trunc'

    @staticmethod
    def forward(ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return torch.trunc(x)

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        return g.op(f'{DOMAIN_STRING}::Trunc',
                    x, scale, zero_point, input_bit_width, output_bit_width,
                    rounding_mode_s=rounding_mode)


class ConvFunc(autograd.Function):
    name = 'Conv'

    @staticmethod
    def forward(ctx, x, quant_input, layer, dilations, group, kernel_shape, pads, strides):
        return layer.conv(x)

    @staticmethod
    def symbolic(g, x, quant_input, layer, dilations, group, kernel_shape, pads, strides):
        return g.op(f'{DOMAIN_STRING}::Conv', x, quant_input, 
                        dilations_i=dilations, 
                        group_i=group, 
                        kernel_shape_i=kernel_shape, 
                        pads_i=pads, 
                        strides_i=strides)


class RoundFunc(autograd.Function):
    name = 'Round'

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def symbolic(g, x):
        return g.op(f'{DOMAIN_STRING}::Round', x)


def get_quant_func(bit_width):
    if bit_width == 1:
        return BinaryQuantFunc
    return QuantFunc

def register_custom_ops():
    for func in HAWQ_FUNCS:
        register_custom_op_symbolic(f'{DOMAIN_STRING}::{func.name}', func.symbolic, 1)


HAWQ_FUNCS = [
    BinaryQuantFunc, 
    QuantFunc, 
    TruncFunc,
    RoundFunc,
    ConvFunc
]
