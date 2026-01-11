# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Exp演算: 指数関数
# Exp operation: Exponential
@script()
def exp_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Exp(input)

model_proto = exp_model.to_model_proto()
onnx.save(model_proto, '25_exp.onnx')
print("saved: 25_exp.onnx")
