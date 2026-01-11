# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Equal演算: 等価比較
# Equal operation: Equality comparison
@script()
def equal_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> BOOL[1, 3, 32, 32]:
    return op.Equal(input1, input2)

model_proto = equal_model.to_model_proto()
onnx.save(model_proto, '42_equal.onnx')
print("saved: 42_equal.onnx")
