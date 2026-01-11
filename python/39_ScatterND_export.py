# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ScatterND演算: 多次元インデックスでの要素更新
# ScatterND operation: Scattering elements by multi-dimensional indices
indices_data = np.array([[[0]], [[2]]], dtype=np.int64)
updates_data = np.ones((2, 4, 4), dtype=np.float32)

@script()
def scatter_nd_model(input: FLOAT[4, 4, 4], indices: INT64[2, 1, 1], updates: FLOAT[2, 4, 4]) -> FLOAT[4, 4, 4]:
    return op.ScatterND(input, indices, updates)

model_proto = scatter_nd_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(indices_data, name='indices'))
model_proto.graph.initializer.append(numpy_helper.from_array(updates_data, name='updates'))
onnx.save(model_proto, '39_scatter_nd.onnx')
print("saved: 39_scatter_nd.onnx")
