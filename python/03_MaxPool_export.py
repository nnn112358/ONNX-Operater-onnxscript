# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# MaxPool演算: 最大値プーリング
# MaxPool operation: Max pooling
@script()
def maxpool_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 16, 16]:
    return op.MaxPool(input, kernel_shape=[2, 2], strides=[2, 2])

model_proto = maxpool_model.to_model_proto()
onnx.save(model_proto, '03_maxpool.onnx')
print("saved: 03_maxpool.onnx")
