# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Unsqueeze演算: 新しい次元を追加
# Unsqueeze operation: Adding new dimensions
axes_data = np.array([2], dtype=np.int64)

@script()
def unsqueeze_model(input: FLOAT[1, 3, 32], axes: INT64[1]) -> FLOAT[1, 3, 1, 32]:
    return op.Unsqueeze(input, axes)

model_proto = unsqueeze_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(axes_data, name='axes'))
onnx.save(model_proto, '33_unsqueeze.onnx')
print("saved: 33_unsqueeze.onnx")
