# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Tanh演算: ハイパボリックタンジェント
# Tanh operation: Hyperbolic tangent
@script()
def tanh_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Tanh(input)

model_proto = tanh_model.to_model_proto()
onnx.save(model_proto, '16_tanh.onnx')
print("saved: 16_tanh.onnx")
