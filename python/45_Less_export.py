# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# And演算: 論理積
# And operation: Logical AND
@script()
def and_model(input1: BOOL[1, 3, 32, 32], input2: BOOL[1, 3, 32, 32]) -> BOOL[1, 3, 32, 32]:
    return op.And(input1, input2)

model_proto = and_model.to_model_proto()
onnx.save(model_proto, '45_and.onnx')
print("saved: 45_and.onnx")
