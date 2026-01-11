# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# LeakyReLU演算: リーキー正規化線形ユニット
# LeakyReLU operation: Leaky Rectified Linear Unit
@script()
def leakyrelu_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.LeakyRelu(input, alpha=0.01)

model_proto = leakyrelu_model.to_model_proto()
onnx.save(model_proto, '10_leakyrelu.onnx')
print("saved: 10_leakyrelu.onnx")
