# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ReduceSum演算: 軸方向の縮約
# ReduceSum operation: Reduction along axis
axes_data = np.array([1], dtype=np.int64)

@script()
def reducesum_model(input: FLOAT[1, 3, 32, 32], axes: INT64[1]) -> FLOAT[1, 1, 32, 32]:
    return op.ReduceSum(input, axes, keepdims=1)

model_proto = reducesum_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(axes_data, name='axes'))
onnx.save(model_proto, '47_reducesum.onnx')
print("saved: 47_reducesum.onnx")
