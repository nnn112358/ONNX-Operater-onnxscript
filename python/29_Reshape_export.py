# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Flatten演算: テンソルの平坦化
# Flatten operation: Flattening tensor
@script()
def flatten_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3072]:
    return op.Flatten(input, axis=1)

model_proto = flatten_model.to_model_proto()
onnx.save(model_proto, '29_flatten.onnx')
print("saved: 29_flatten.onnx")
