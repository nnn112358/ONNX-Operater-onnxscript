# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Pow演算: べき乗
# Pow operation: Power
@script()
def pow_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Pow(input1, input2)

model_proto = pow_model.to_model_proto()
onnx.save(model_proto, '23_pow.onnx')
print("saved: 23_pow.onnx")
