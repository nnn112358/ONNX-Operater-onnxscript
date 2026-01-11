# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Log演算: 自然対数
# Log operation: Natural logarithm
@script()
def log_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Log(input)

model_proto = log_model.to_model_proto()
onnx.save(model_proto, '26_log.onnx')
print("saved: 26_log.onnx")
