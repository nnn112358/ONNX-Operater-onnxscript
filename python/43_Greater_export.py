# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Greater演算: 大小比較
# Greater operation: Greater than comparison
@script()
def greater_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> BOOL[1, 3, 32, 32]:
    return op.Greater(input1, input2)

model_proto = greater_model.to_model_proto()
onnx.save(model_proto, '43_greater.onnx')
print("saved: 43_greater.onnx")
