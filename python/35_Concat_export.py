# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Concat演算: テンソルの連結
# Concat operation: Concatenating tensors
@script()
def concat_model(input1: FLOAT[1, 2, 32, 32], input2: FLOAT[1, 2, 32, 32]) -> FLOAT[1, 4, 32, 32]:
    return op.Concat(input1, input2, axis=1)

model_proto = concat_model.to_model_proto()
onnx.save(model_proto, '35_concat.onnx')
print("saved: 35_concat.onnx")
