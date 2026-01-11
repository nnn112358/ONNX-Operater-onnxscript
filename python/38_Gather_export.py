# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Gather演算: インデックスによる要素収集
# Gather operation: Gathering elements by indices
indices_data = np.array([0, 2], dtype=np.int64)

@script()
def gather_model(input: FLOAT[1, 4, 32, 32], indices: INT64[2]) -> FLOAT[1, 2, 32, 32]:
    return op.Gather(input, indices, axis=1)

model_proto = gather_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(indices_data, name='indices'))
onnx.save(model_proto, '38_gather.onnx')
print("saved: 38_gather.onnx")
