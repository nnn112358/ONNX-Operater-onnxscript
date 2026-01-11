# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Reshape演算: テンソルの形状変更
# Reshape operation: Reshaping tensor
shape_data = np.array([1, -1], dtype=np.int64)

@script()
def reshape_model(input: FLOAT[1, 3, 32, 32], shape: INT64[2]) -> FLOAT[1, 3072]:
    return op.Reshape(input, shape)

model_proto = reshape_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(shape_data, name='shape'))
onnx.save(model_proto, '18_reshape.onnx')
print("saved: 18_reshape.onnx")
