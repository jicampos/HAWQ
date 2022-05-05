import onnx

if __name__ == '__main__':
    
    onnx_model = onnx.load('Q_JetTagger_04222022164654.onnx')

    # hls_model = hls4ml.converters.convert_from_onnx_model(onnx_model)
    # print(hls_model)
    
    graph = onnx_model.graph

    for node in graph.node:
        print(node.op_type)

        # node inputs
        for idx, node_input_name in enumerate(node.input):
            print(idx, node_input_name)
        # node outputs
        for idx, node_output_name in enumerate(node.output):
            print(idx, node_output_name)
