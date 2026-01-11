# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# SpaceToDepth演算: 空間次元をチャンネル次元に変換
# SpaceToDepth operation: Converting spatial dimensions to channel dimension
@script()
def space_to_depth_model(input: FLOAT[1, 3, 32, 32]) -> FLOAT[1, 12, 16, 16]:
    return op.SpaceToDepth(input, blocksize=2)

model_proto = space_to_depth_model.to_model_proto()
onnx.save(model_proto, '58_space_to_depth.onnx')
print("saved: 58_space_to_depth.onnx")
