# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# GlobalAveragePool演算: グローバル平均プーリング
# GlobalAveragePool operation: Global average pooling
@script()
def globalavgpool_model(input: FLOAT[1, 512, 7, 7]) -> FLOAT[1, 512, 1, 1]:
    return op.GlobalAveragePool(input)

model_proto = globalavgpool_model.to_model_proto()
onnx.save(model_proto, '05_globalavgpool.onnx')
print("saved: 05_globalavgpool.onnx")
