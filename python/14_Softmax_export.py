# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Softmax演算: ソフトマックス関数
# Softmax operation: Softmax function
@script()
def softmax_model(input: FLOAT[4, 10]) -> FLOAT[4, 10]:
    return op.Softmax(input, axis=-1)

model_proto = softmax_model.to_model_proto()
onnx.save(model_proto, '14_softmax.onnx')
print("saved: 14_softmax.onnx")
