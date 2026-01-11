# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Div演算: 要素ごとの除算
# Div operation: Element-wise division
@script()
def div_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Div(input1, input2)

model_proto = div_model.to_model_proto()
onnx.save(model_proto, '22_div.onnx')
print("saved: 22_div.onnx")
