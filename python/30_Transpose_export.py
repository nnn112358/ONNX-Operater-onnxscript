# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Transpose演算: 軸の入れ替え
# Transpose operation: Transposing axes
@script()
def transpose_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 32, 32, 3]:
    return op.Transpose(input, perm=[0, 2, 3, 1])

model_proto = transpose_model.to_model_proto()
onnx.save(model_proto, '30_transpose.onnx')
print("saved: 30_transpose.onnx")
