# Export HAWQ to QONNX

Translate models to Quantized ONNX ([QONNX](https://github.com/fastmachinelearning/qonnx)), an extension of the Open Neural Network Exchange ([ONNX](https://github.com/onnx/onnx)) format for arbitrary-precision quantized neural networks.

### Export Model
```python 
from utils.export import ExportManager

...

manager = ExportManager(hawq_model)
manager.export(
    torch.randn([1, 16]),   # dummy input for tracing 
    "hawq2qonnx_model.onnx"
)
```

### Model Visualization
The [Netron](https://netron.app/) visualizer is recommended for QONNX models.

### Execute ONNX graph with QONNX operators
```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

...

qonnx_model = ModelWrapper('hawq2qonnx_model.onnx')

input_dict = {"global_in": X_test[0]}

output_dict = execute_onnx(qonnx_model, input_dict)
```