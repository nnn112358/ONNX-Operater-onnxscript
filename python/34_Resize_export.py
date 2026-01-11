# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Resize演算: テンソルのサイズ変更
# Resize operation: Resizing tensor
roi_data = np.array([], dtype=np.float32)
scales_data = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

@script()
def resize_model(input: FLOAT[1, 3, 16, 16], roi: FLOAT[0], scales: FLOAT[4]) -> FLOAT[1, 3, 32, 32]:
    return op.Resize(input, roi, scales, mode='linear')

model_proto = resize_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(roi_data, name='roi'))
model_proto.graph.initializer.append(numpy_helper.from_array(scales_data, name='scales'))
onnx.save(model_proto, '34_resize.onnx')
print("saved: 34_resize.onnx")
