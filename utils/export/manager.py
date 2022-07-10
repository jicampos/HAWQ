import copy
import logging
import warnings

import torch
import torch.nn as nn
from torch._C import ListType, OptionalType

import onnx
import qonnx
from qonnx.util.cleanup import cleanup
import onnxoptimizer

from .export_modules import model_info
from .function import register_custom_ops, domain_info


from .export_modules import (
    ExportQonnxQuantAct,
    ExportQonnxQuantLinear,
    ExportQonnxQuantConv2d,
    ExportQonnxQuantAveragePool2d,
    ExportQonnxQuantBnConv2d,
)
from ..quantization_utils.quant_modules import (
    QuantAct,
    QuantLinear,
    QuantBnConv2d,
    QuantAveragePool2d,
    QuantConv2d,
)

SET_EXPORT_MODE = (
    ExportQonnxQuantAct,
    ExportQonnxQuantLinear,
    ExportQonnxQuantConv2d,
    ExportQonnxQuantAveragePool2d,
    ExportQonnxQuantBnConv2d,
)


# ------------------------------------------------------------
class ExportManager(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        assert model is not None, "Model is not initialized"

        self.copy_model(model)
        self.replace_layers()
        self.init_tracker()

    def predict(self, x):
        self.set_export_mode("enable")
        export_pred = self.export_model(x)
        return export_pred

    def forward(self, x):
        self.set_export_mode("disable")
        hawq_pred = self.export_model(x)
        self.set_export_mode("enable")
        export_pred = self.export_model(x)
        return export_pred, hawq_pred

    def copy_model(self, model):
        try:
            self.export_model = copy.deepcopy(model)
        except Exception as e:
            logging.error(e)
            raise Exception(e)

    def replace_layers(self):
        for param in self.export_model.parameters():
            param.requires_grad_(False)

        for name in dir(self.export_model):
            layer = getattr(self.export_model, name)
            onnx_export_layer = None
            if isinstance(layer, QuantAct):
                onnx_export_layer = ExportQonnxQuantAct(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantLinear):
                onnx_export_layer = ExportQonnxQuantLinear(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantConv2d):
                onnx_export_layer = ExportQonnxQuantConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantBnConv2d):
                onnx_export_layer = ExportQonnxQuantBnConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantAveragePool2d):
                onnx_export_layer = ExportQonnxQuantAveragePool2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, nn.Sequential):
                # no nn.ModuleList?
                self.replace_layers(layer)
            # track changes
            if onnx_export_layer is not None:
                model_info["transformed"][layer] = onnx_export_layer
        for param in self.export_model.parameters():
            param.requires_grad_(True)

    def init_tracker(self):
        """Track output of export and original layers"""
        self.tracker = dict()
        self.tracker["dense_out"] = dict()
        self.tracker["dense_out_export_mode"] = dict()
        self.tracker["quant_out"] = dict()
        self.tracker["quant_out_export_mode"] = dict()
        self.tracker["transformed"] = dict()
        self.tracker["act_scaling_factor"] = dict()
        self.tracker["conv_scaling_factor"] = dict()
        self.tracker["convbn_scaling_factor"] = dict()

    @staticmethod
    def enable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = True

    @staticmethod
    def disable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = False

    def set_export_mode(self, export_mode):
        if export_mode == "enable":
            self.export_model.apply(self.enable_export)
        else:
            self.export_model.apply(self.disable_export)

    def optimize_onnx_model(self, model_path):
        onnx_model = onnxoptimizer.optimize(
            onnx.load_model(model_path), passes=["extract_constant_to_initializer"]
        )
        cleanup(onnx_model, out_file=model_path)

    def gen_filename(self):
        from datetime import datetime

        now = datetime.now()  # current date and time
        date_time = now.strftime("%m%d%Y_%H%M%S")
        if "Jet" in self.export_model._get_name():
            filename = f"hawq2qonnx_jet_{date_time}.onnx"
        elif "MNIST" in self.export_model._get_name():
            filename = f"hawq2qonnx_mnist_{date_time}.onnx"
        else:
            filename = f"hawq2qonnx_{date_time}.onnx"
        return filename

    def export(self, x, filename=None, save=True):
        assert x is not None, "Input x is not initialized"
        assert type(x) is torch.Tensor, "Input x must be a torch.Tensor"

        if filename is None:
            filename = self.gen_filename()
        if len(x) > 1:
            logging.info("Only [1, ?] dimensions are supported. Selecting first.")
            x = x[0].view(1, -1)
        register_custom_ops()

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # collect scaling factors for onnx nodes
                self.set_export_mode("disable")
                _ = self.export_model(x)
                # export with collected values
                self.set_export_mode("enable")
                if save:
                    print("Exporting model...")
                    torch.onnx.export(
                        model=self.export_model,
                        args=x,
                        f=filename,
                        opset_version=11,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                        custom_opsets={domain_info["name"]: 1},
                    )
                    print(f"Model saved: {filename}")
                    self.optimize_onnx_model(filename)
