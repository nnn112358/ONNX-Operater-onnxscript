# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Clip演算: 値の範囲制限
# Clip operation: Clipping values to range
min_data = np.array([0.0], dtype=np.float32)
max_data = np.array([6.0], dtype=np.float32)

@script()
def clip_model(input: FLOAT[1, 3, 32, 32], min_val: FLOAT[1], max_val: FLOAT[1]) -> FLOAT[1, 3, 32, 32]:
    return op.Clip(input, min_val, max_val)

model_proto = clip_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(min_data, name='min_val'))
model_proto.graph.initializer.append(numpy_helper.from_array(max_data, name='max_val'))
onnx.save(model_proto, '17_clip.onnx')
print("saved: 17_clip.onnx")
