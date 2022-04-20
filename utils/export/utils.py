from ..quantization_utils.quant_modules import QuantAct, QuantLinear

def enable_export(module):
    if isinstance(module, QuantAct) or isinstance(module, QuantLinear):
        module.export_mode = True

def disable_export(module):
    if type(module) == QuantAct or type(module) == QuantLinear:
        module.export_mode = False 

def set_export_mode(module, export_mode):
    if export_mode == 'enable':
        module.apply(enable_export)
    else:
        module.apply(disable_export)

