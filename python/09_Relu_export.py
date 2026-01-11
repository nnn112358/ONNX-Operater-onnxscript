# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ReLU演算: 正規化線形ユニット
# ReLU operation: Rectified Linear Unit
@script()
def relu_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Relu(input)

model_proto = relu_model.to_model_proto()
onnx.save(model_proto, '09_relu.onnx')
print("saved: 09_relu.onnx")
