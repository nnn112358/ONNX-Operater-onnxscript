# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# DepthToSpace演算: チャンネル次元を空間次元に変換
# DepthToSpace operation: Converting channel dimension to spatial dimensions
@script()
def depth_to_space_model(input: FLOAT[1, 12, 16, 16]) -> FLOAT[1, 3, 32, 32]:
    return op.DepthToSpace(input, blocksize=2)

model_proto = depth_to_space_model.to_model_proto()
onnx.save(model_proto, '59_depth_to_space.onnx')
print("saved: 59_depth_to_space.onnx")
