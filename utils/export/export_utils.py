import struct

import onnx
import qonnx
from qonnx.util.cleanup import cleanup
from qonnx.util.to_channels_last import to_channels_last
import onnxoptimizer


def optimize_onnx_model(model_path):
    onnx_model = onnxoptimizer.optimize(
        onnx.load_model(model_path), passes=["extract_constant_to_initializer"]
    )
    cleanup(onnx_model, out_file=model_path)
    to_channels_last(model_path, out_file=model_path)
    onnx_model = onnx.load(model_path)
    merge_mul_div_nodes(onnx_model)
    onnx.save_model(onnx_model, model_path)
    cleanup(onnx_model, out_file=model_path)


def merge_mul_div_nodes(model):
    for node in model.graph.node:
        if node.op_type == "Mul":
            next_node = find_next_op(model, node.output[0])
            merge_nodes(model, node, next_node)


def find_next_op(model, target):
    for node in model.graph.node:
        node_inputs = node.input
        if target in node_inputs:
            return node


def merge_nodes(model, mul_node, next_node):
    if next_node.op_type != "Div":
        return

    print(f"Merging {next_node.name} with {mul_node.name}")
    mul_scalar = 0
    div_scalar = 0
    mul_param = None
    div_param = None
    mul_param_name = mul_node.name + "_param0"
    div_param_name = next_node.name + "_param0"

    for param in model.graph.initializer:
        if param.name == mul_param_name:
            mul_param = param
            mul_scalar = struct.unpack("f", param.raw_data)[0]
        if param.name == div_param_name:
            div_param = param
            div_scalar = struct.unpack("f", param.raw_data)[0]

    new_mul_data = mul_scalar / div_scalar
    print(f"New mul parameter {new_mul_data}")
    mul_param.raw_data = struct.pack("f", new_mul_data)

    remove_div_node(model, mul_node, next_node)


def remove_div_node(model, mul_node, div_node):
    mul_node_output = mul_node.output[0]

    next_node_after_div = find_next_op(model, div_node.output[0])
    node_inputs = next_node_after_div.input
    for idx, node_in in enumerate(node_inputs):
        if node_in == div_node.output[0]:
            print(
                f"Changing node {next_node_after_div.name} param {node_in} to {mul_node_output}"
            )
            next_node_after_div.input[idx] = mul_node_output

    print(f"Removing node {div_node.name}")
    model.graph.node.remove(div_node)


def gen_filename():
    from datetime import datetime

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S")
    if "Jet" in export_model._get_name():
        filename = f"hawq2qonnx_jet_{date_time}.onnx"
    elif "MNIST" in export_model._get_name():
        filename = f"hawq2qonnx_mnist_{date_time}.onnx"
    else:
        filename = f"hawq2qonnx_{date_time}.onnx"
    return filename
