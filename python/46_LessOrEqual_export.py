# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Or演算: 論理和
# Or operation: Logical OR
@script()
def or_model(input1: BOOL[1, 3, 32, 32], input2: BOOL[1, 3, 32, 32]) -> BOOL[1, 3, 32, 32]:
    return op.Or(input1, input2)

model_proto = or_model.to_model_proto()
onnx.save(model_proto, '46_or.onnx')
print("saved: 46_or.onnx")
