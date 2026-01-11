# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# MatMul演算: 行列積
# MatMul operation: Matrix multiplication
@script()
def matmul_model(input1: FLOAT[4, 8], input2: FLOAT[8, 16]) -> FLOAT[4, 16]:
    return op.MatMul(input1, input2)

model_proto = matmul_model.to_model_proto()
onnx.save(model_proto, '40_matmul.onnx')
print("saved: 40_matmul.onnx")
