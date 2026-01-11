# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# GRU演算: ゲート付き回帰ユニット
# GRU operation: Gated Recurrent Unit
W_data = np.random.randn(1, 48, 16).astype(np.float32)
R_data = np.random.randn(1, 48, 16).astype(np.float32)
B_data = np.random.randn(1, 96).astype(np.float32)

@script()
def gru_model(input: FLOAT[5, 4, 16], W: FLOAT[1, 48, 16], R: FLOAT[1, 48, 16], B: FLOAT[1, 96]) -> FLOAT[5, 1, 4, 16]:
    Y, Y_h = op.GRU(input, W, R, B, hidden_size=16)
    return Y

model_proto = gru_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(W_data, name='W'))
model_proto.graph.initializer.append(numpy_helper.from_array(R_data, name='R'))
model_proto.graph.initializer.append(numpy_helper.from_array(B_data, name='B'))
onnx.save(model_proto, '08_gru.onnx')
print("saved: 08_gru.onnx")
