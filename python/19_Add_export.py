# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Add演算: 要素ごとの加算
# Add operation: Element-wise addition
@script()
def add_model(input1: FLOAT[1, 3, 32, 32], input2: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Add(input1, input2)

model_proto = add_model.to_model_proto()
onnx.save(model_proto, '19_add.onnx')
print("saved: 19_add.onnx")
