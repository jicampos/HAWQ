"""
Exporting the JetTagger model in FINN/ONNX format.
"""

import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
from utils import q_jettagger_model
from utils.export.utils import set_export_mode

from utils.quantization_utils.function import BrevitasQuantFn

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("::MyReLu", BrevitasQuantFn.symbolic, 1)


# ------------------------------------------------------------
class QuantActFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def symbolic(g, x):
        return g.op("Quant", x, g.op("Constant", value_t=torch.tensor(1.23, dtype=torch.float)))

register_custom_op_symbolic("::QuantAct", QuantActFunc.symbolic, 1)

# ------------------------------------------------------------
class MyQuantAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.export_mode = True
        self.relu = torch.nn.ReLU()

    def forward(self, x, act_scaling_factor=None, weight_scaling_factor=None):
        if self.export_mode:
            return (QuantActFunc.apply(x), act_scaling_factor)
        else:
            return self.relu(x)

# ------------------------------------------------------------
class MyQuantLinear(torch.nn.Module):
    def __init__(self, input_shape, out_shape) -> None:
        super().__init__()
        self.export_mode = True
        self.fc = torch.nn.Linear(input_shape, out_shape)

    def forward(self, x, act_scaling_factor=None, weight_scaling_factor=None):
        if self.export_mode:
            return (self.fc(x), act_scaling_factor)
        else:
            return self.fc(x)


def replace_with_nn(model):
    model = model.features
    for param in model.parameters():
        param.requires_grad_(False)
    
    from utils.quantization_utils.quant_modules import QuantAct, QuantLinear
    for name, layer in enumerate(model):
        if isinstance(layer, QuantAct):
            print('quantact')
            model[name] = MyQuantAct()
        if isinstance(layer, QuantLinear):
            print('quantlinear')
            model[name] = MyQuantLinear(layer.weight.shape[1], layer.weight.shape[0])

def replace_nn_apply(model):
    for param in model.parameters():
        param.requires_grad_(False)
    
    from utils.quantization_utils.quant_modules import QuantAct, QuantLinear
    if isinstance(model, QuantAct):
        print('quantact')
    if isinstance(model, QuantLinear):
        print('quantlinear')

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
    

    replace_with_nn(model)
    print(model)

    setattr(model, 'quant_input', MyQuantAct())
    setattr(model, 'quant_act1', MyQuantAct())
    setattr(model, 'quant_act2', MyQuantAct())
    setattr(model, 'quant_act3', MyQuantAct())
    setattr(model, 'quant_act4', MyQuantAct())
    print(model)

    model(x)

    register_custom_op_symbolic("::Quant", QuantActFunc.symbolic, 1)
    torch.onnx.export(model, x, 'hawq_replaced_layers_v3.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK) 
    
    # print(list(model.modules())[1:])
    # # print(model(x))

    
    # keys = []
    # for name, value in model.named_parameters():
    #     keys.append(name)

    # # with torch.no_grad():
    # #     getattr(model, keys[0].split('.')[0]).weight.fill_(0.)  # split key to only get 'fc1'

    # print(keys)
    # # replace_with_nn(model)
    # model.apply(replace_nn_apply)

    bp = 0


