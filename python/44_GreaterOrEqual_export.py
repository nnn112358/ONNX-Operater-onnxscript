# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Less演算: 小大比較
# Less operation: Less than comparison
@script()
def less_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> BOOL[1, 3, 32, 32]:
    return op.Less(input1, input2)

model_proto = less_model.to_model_proto()
onnx.save(model_proto, '44_less.onnx')
print("saved: 44_less.onnx")
