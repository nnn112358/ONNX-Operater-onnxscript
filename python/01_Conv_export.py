# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Conv演算: 2D畳み込み
# Conv operation: 2D Convolution
weight_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
bias_data = np.random.randn(16).astype(np.float32)

@script()
def conv_model(input: FLOAT[1, 3, 32, 32], weight: FLOAT[16, 3, 3, 3], bias: FLOAT[16]) -> FLOAT[1, 16, 32, 32]:
    return op.Conv(input, weight, bias, kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])

model_proto = conv_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(weight_data, name='weight'))
model_proto.graph.initializer.append(numpy_helper.from_array(bias_data, name='bias'))
onnx.save(model_proto, '01_conv.onnx')
print("saved: 01_conv.onnx")
