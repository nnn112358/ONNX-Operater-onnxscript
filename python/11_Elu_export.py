# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ELU演算: 指数線形ユニット
# ELU operation: Exponential Linear Unit
@script()
def elu_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Elu(input, alpha=1.0)

model_proto = elu_model.to_model_proto()
onnx.save(model_proto, '11_elu.onnx')
print("saved: 11_elu.onnx")
