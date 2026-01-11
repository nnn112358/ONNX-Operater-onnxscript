# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# PReLU演算: パラメトリックReLU
# PReLU operation: Parametric ReLU
slope_data = np.array([0.25], dtype=np.float32)

@script()
def prelu_model(input: FLOAT[1, 3, 32, 32], slope: FLOAT[1]) -> FLOAT[1, 3, 32, 32]:
    return op.PRelu(input, slope)

model_proto = prelu_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(slope_data, name='slope'))
onnx.save(model_proto, '12_prelu.onnx')
print("saved: 12_prelu.onnx")
