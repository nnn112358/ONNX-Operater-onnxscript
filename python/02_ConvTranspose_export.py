# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ConvTranspose演算: 転置畳み込み（デコンボリューション）
# ConvTranspose operation: Transposed convolution (deconvolution)
weight_data = np.random.randn(16, 3, 3, 3).astype(np.float32)
bias_data = np.random.randn(3).astype(np.float32)

@script()
def convtranspose_model(input: FLOAT[1, 16, 16, 16], weight: FLOAT[16, 3, 3, 3], bias: FLOAT[3]) -> FLOAT[1, 3, 32, 32]:
    return op.ConvTranspose(input, weight, bias, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])

model_proto = convtranspose_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(weight_data, name='weight'))
model_proto.graph.initializer.append(numpy_helper.from_array(bias_data, name='bias'))
onnx.save(model_proto, '02_conv_transpose.onnx')
print("saved: 02_conv_transpose.onnx")
