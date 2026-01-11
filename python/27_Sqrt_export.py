# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Sqrt演算: 平方根
# Sqrt operation: Square root
@script()
def sqrt_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Sqrt(input)

model_proto = sqrt_model.to_model_proto()
onnx.save(model_proto, '27_sqrt.onnx')
print("saved: 27_sqrt.onnx")
