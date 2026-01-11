# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Slice演算: テンソルの切り出し
# Slice operation: Slicing tensor
starts_data = np.array([1], dtype=np.int64)
ends_data = np.array([3], dtype=np.int64)
axes_data = np.array([1], dtype=np.int64)

@script()
def slice_model(input: FLOAT[1, 4, 32, 32], starts: INT64[1], ends: INT64[1], axes: INT64[1]) -> FLOAT[1, 2, 32, 32]:
    return op.Slice(input, starts, ends, axes)

model_proto = slice_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(starts_data, name='starts'))
model_proto.graph.initializer.append(numpy_helper.from_array(ends_data, name='ends'))
model_proto.graph.initializer.append(numpy_helper.from_array(axes_data, name='axes'))
onnx.save(model_proto, '37_slice.onnx')
print("saved: 37_slice.onnx")
