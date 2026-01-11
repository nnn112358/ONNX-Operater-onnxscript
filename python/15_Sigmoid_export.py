# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Sigmoid演算: シグモイド関数
# Sigmoid operation: Sigmoid function
@script()
def sigmoid_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Sigmoid(input)

model_proto = sigmoid_model.to_model_proto()
onnx.save(model_proto, '15_sigmoid.onnx')
print("saved: 15_sigmoid.onnx")
