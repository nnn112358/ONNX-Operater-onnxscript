# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Swish演算: x * sigmoid(x)
# Swish operation: x * sigmoid(x)
@script()
def swish_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Mul(input, op.Sigmoid(input))

model_proto = swish_model.to_model_proto()
onnx.save(model_proto, '13_swish.onnx')
print("saved: 13_swish.onnx")
