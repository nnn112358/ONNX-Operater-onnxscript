# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Neg演算: 符号反転
# Neg operation: Negation
@script()
def neg_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 32, 32]:
    return op.Neg(input)

model_proto = neg_model.to_model_proto()
onnx.save(model_proto, '24_neg.onnx')
print("saved: 24_neg.onnx")
