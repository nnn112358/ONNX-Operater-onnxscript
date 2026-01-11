# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Pad演算: パディング
# Pad operation: Padding
pads_data = np.array([0, 0, 2, 2, 0, 0, 2, 2], dtype=np.int64)

@script()
def pad_model(input: FLOAT[1, 3, 32, 32], pads: INT64[8]) -> FLOAT[1, 3, 36, 36]:
    return op.Pad(input, pads)

model_proto = pad_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(pads_data, name='pads'))
onnx.save(model_proto, '57_pad.onnx')
print("saved: 57_pad.onnx")
