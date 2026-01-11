# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# AveragePool演算: 平均値プーリング
# AveragePool operation: Average pooling
@script()
def avgpool_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 3, 16, 16]:
    return op.AveragePool(input, kernel_shape=[2, 2], strides=[2, 2])

model_proto = avgpool_model.to_model_proto()
onnx.save(model_proto, '04_avgpool.onnx')
print("saved: 04_avgpool.onnx")
