# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Squeeze演算: サイズ1の次元を削除
# Squeeze operation: Removing dimensions of size 1
axes_data = np.array([2], dtype=np.int64)

@script()
def squeeze_model(input: FLOAT[1, 3, 1, 32], axes: INT64[1]) -> FLOAT[1, 3, 32]:
    return op.Squeeze(input, axes)

model_proto = squeeze_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(axes_data, name='axes'))
onnx.save(model_proto, '32_squeeze.onnx')
print("saved: 32_squeeze.onnx")
