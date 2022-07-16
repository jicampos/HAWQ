import logging

import torch
import torch.nn as nn
from torch._C import ListType, OptionalType

from .function import get_quant_func
from .function import TruncFunc, RoundFunc, ConvFunc

from ..quantization_utils.quant_modules import (
    QuantAct,
    QuantDropout,
    QuantLinear,
    QuantBnConv2d,
)
from ..quantization_utils.quant_modules import (
    QuantMaxPool2d,
    QuantAveragePool2d,
    QuantConv2d,
)

SUPPORTED_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantMaxPool2d,
    QuantAveragePool2d,
    QuantDropout,
    QuantBnConv2d,
)

model_info = dict()
model_info["dense_out"] = dict()
model_info["dense_out_export_mode"] = dict()
model_info["quant_in"] = dict()
model_info["quant_out"] = dict()
model_info["quant_out_export_mode"] = dict()
model_info["transformed"] = dict()
model_info["act_scaling_factor"] = dict()
model_info["conv_scaling_factor"] = dict()
model_info["convbn_scaling_factor"] = dict()

# ------------------------------------------------------------
class ExportQonnxQuantAct(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.export_mode = False
        self.init_parameters()

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}(scale={self.scale.detach().item()}, zero_point=0, bitwidth={self.layer.activation_bit},"
            + f" full_precision_flag={self.layer.full_precision_flag}, quant_mode={self.layer.quant_mode})"
        )
        return s

    def init_parameters(self):
        if self.layer.full_precision_flag:
            self.bit_width = 32
        else:
            self.bit_width = self.layer.activation_bit
        self.n = 2 ** (self.bit_width - 1) - 1
        self.clamp_min = -self.n - 1
        self.clamp_max = self.n

        self.scale = (
            self.layer.act_scaling_factor.clone().detach().requires_grad_(False)
        )
        self.node_inputs = (
            torch.tensor(self.scale.item(), dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.bit_width, dtype=torch.float32),  # bit width
        )
        self.node_attributes = (
            int(1 if self.layer.quant_mode == "symmetric" else 0),  # signed
            int(0),  # narrow range
            "ROUND",  # rounding mode
        )

    def update_node_inputs(self, pre_act_scaling_factor):
        from ..quantization_utils.quant_utils import batch_frexp

        new_scale = pre_act_scaling_factor / self.scale
        m, e = batch_frexp(new_scale)
        new_scale = m / (2.0**e)
        new_scale = 1 / new_scale
        self.node_inputs = (
            torch.tensor(new_scale.item(), dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.bit_width, dtype=torch.float32),  # bit width
        )

    def compute_quant_act(self, z_int, pre_act_scaling_factor):
        from ..quantization_utils.quant_utils import batch_frexp

        new_scale = pre_act_scaling_factor / self.scale
        m, e = batch_frexp(new_scale)
        m = m.type(torch.float32)
        e = e.type(torch.float32)
        output = z_int * m
        output = output / (2.0**e)
        output = torch.round(output)
        return torch.clamp(output, self.clamp_min, self.clamp_max)

    def pre_quant_scale(self, x, pre_act_scaling_factor):
        if pre_act_scaling_factor.item() != 1:
            x = x / pre_act_scaling_factor.item()
        # x = torch.round(x)
        return x, pre_act_scaling_factor

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        pre_weight_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        identity_weight_scaling_factor=None,
    ):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        if pre_act_scaling_factor is None:
            pre_act_scaling_factor = torch.tensor([1.0], dtype=torch.float32)

        if self.export_mode:
            x, pre_act_scaling_factor = self.pre_quant_scale(x, pre_act_scaling_factor)
            quant_node = get_quant_func(self.bit_width)
            x = quant_node.apply(x, *self.node_inputs, *self.node_attributes)
            # x = self.compute_quant_act(x, pre_act_scaling_factor)
            model_info["quant_out_export_mode"][self.layer] = x
            return (x, self.scale)
        else:
            # self.update_node_inputs(pre_act_scaling_factor)
            x, act_scaling_factor = self.layer(
                x,
                pre_act_scaling_factor,
                pre_weight_scaling_factor,
                identity,
                identity_scaling_factor,
                identity_weight_scaling_factor,
            )
            model_info["act_scaling_factor"][self.layer] = act_scaling_factor
            model_info["quant_out"][self.layer] = x
            return (x, act_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantLinear(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.export_mode = False

        self.has_bias = hasattr(layer, "bias")
        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        self.fc = torch.nn.Linear(in_features, out_features, self.has_bias)
        self.fc.weight.data = torch.transpose(layer.weight_integer, 0, 1)
        if self.has_bias:
            self.fc.bias.data = layer.bias_integer

        self.scale = self.layer.fc_scaling_factor.clone().requires_grad_(False)
        self.weight_node_inputs = (
            torch.tensor(1, dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.layer.weight_bit, dtype=torch.float32),  # bit width
        )

        if self.has_bias:
            self.bias_node_inputs = (
                torch.tensor(1, dtype=torch.float32),  # scale
                torch.tensor(0, dtype=torch.float32),  # zero point
                torch.tensor(self.layer.bias_bit, dtype=torch.float32),  # bit width
            )

        self.node_attributes = (
            int(1 if self.layer.quant_mode == "symmetric" else 0),  # signed
            int(0),  # narrow range
            "ROUND",  # rounding mode
        )

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}(weight_bit={self.layer.weight_bit},"
            + f" bias_bit={self.layer.bias_bit}, quantize={self.layer.quant_mode})"
        )
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            # x = x / prev_act_scaling_factor.view(1, -1)
            quant_node = get_quant_func(self.weight_node_inputs[2].item())
            weights = quant_node.apply(
                self.fc.weight.data, *self.weight_node_inputs, *self.node_attributes
            )
            x = torch.matmul(x, weights)

            if self.has_bias:
                quant_node = get_quant_func(self.bias_node_inputs[2].item())
                bias = quant_node.apply(
                    self.fc.bias.data, *self.bias_node_inputs, *self.node_attributes
                )
                x = torch.add(x, bias)

            x = torch.round(x)
            bias_scaling_factor = self.scale * prev_act_scaling_factor.clone()
            x = x * bias_scaling_factor.view(1, -1)
            model_info["dense_out_export_mode"][self.layer] = x.clone()
            return x
        else:
            x = self.layer(x, prev_act_scaling_factor)
            model_info["dense_out"][self.layer] = x.clone()
            return x


# ------------------------------------------------------------
class ExportQonnxQuantConv2d(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.export_mode = False

        self.conv_scaling_factor = self.layer.conv_scaling_factor.detach().clone()
        self.quant_args = (
            torch.tensor(1, dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.layer.weight_bit, dtype=torch.float32),  # bit width
            int(1 if layer.quant_mode == "symmetric" else 0),  # signed
            int(0),  # narrow range
            "ROUND",  # rounding mode
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
        elif len(pads) == 2:
            pads = (pads[0], pads[0], pads[1], pads[1])

        strides = self.layer.conv.stride
        if type(strides) != tuple:
            strides = (strides, strides)

        self.conv_args = (dilation, self.layer.conv.groups, kernel_size, pads, strides)

    def __repr__(self):
        s = f"{self.__class__.__name__}()"
        return s

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]
        if x.ndim == 3:
            # add an extra dimension for batch size
            x = x[None]

        bias_scaling_factor = self.conv_scaling_factor.view(
            1, -1
        ) * prev_act_scaling_factor.view(1, -1)
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)
        correct_output_scale = correct_output_scale[0][0][0][0]

        if self.export_mode:
            QuantFunc = get_quant_func(self.layer.weight_bit)
            weights = QuantFunc.apply(self.layer.weight_integer, *self.quant_args)
            return (
                ConvFunc.apply(x, weights, self.layer, *self.conv_args)
                * correct_output_scale,
                self.conv_scaling_factor,
            )
        else:
            x, conv_scaling_factor = self.layer(x, prev_act_scaling_factor)
            model_info["conv_scaling_factor"][self.layer] = conv_scaling_factor
            return (x, conv_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantBnConv2d(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.export_mode = False

        quant_layer = QuantConv2d()
        quant_layer.set_param(self.layer.conv)
        self.export_quant_conv = ExportQonnxQuantConv2d(quant_layer)

        self.bn = torch.nn.BatchNorm2d(
            self.layer.bn.num_features, self.layer.bn.eps, self.layer.bn.momentum
        )
        self.bn.weight = self.layer.bn.weight
        self.bn.bias = self.layer.bn.bias

        self.bn.running_mean = self.layer.bn.running_mean
        self.bn.running_var = self.layer.bn.running_var

    def __repr__(self):
        s = f"{self.__class__.__name__}()"
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
            model_info["convbn_scaling_factor"][self.layer] = convbn_scaling_factor
            return self.export_quant_conv(x, convbn_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantAveragePool2d(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.export_mode = False

        self.trunc_args = (
            torch.tensor(1),  # scale
            torch.tensor(0),  # zero point
            torch.tensor(32),  # input bit width
            torch.tensor(32),  # output bit width
            "ROUND",  # rounding mode
        )

    def __repr__(self):
        s = f"{self.__class__.__name__}()"
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
            x_int = RoundFunc.apply(x_int)
            x_int = self.layer.final_pool(x_int)

            x_int = TruncFunc.apply(x_int + 0.01, *self.trunc_args)

            return (x_int * correct_scaling_factor, correct_scaling_factor)
        else:
            return (self.layer(x), x_scaling_factor)
