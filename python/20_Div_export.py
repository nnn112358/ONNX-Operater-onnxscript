# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Sub演算: 要素ごとの減算
# Sub operation: Element-wise subtraction
@script()
def sub_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Sub(input1, input2)

model_proto = sub_model.to_model_proto()
onnx.save(model_proto, '20_sub.onnx')
print("saved: 20_sub.onnx")
