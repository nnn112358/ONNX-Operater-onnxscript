# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# LayerNormalization演算: 層正規化
# LayerNormalization operation: Layer normalization
scale_data = np.ones(32, dtype=np.float32)
bias_data = np.zeros(32, dtype=np.float32)

@script()
def layernorm_model(input: FLOAT[4, 8, 32], scale: FLOAT[32], bias: FLOAT[32]) -> FLOAT[4, 8, 32]:
    return op.LayerNormalization(input, scale, bias, axis=-1, epsilon=1e-5)

model_proto = layernorm_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(scale_data, name='scale'))
model_proto.graph.initializer.append(numpy_helper.from_array(bias_data, name='bias'))
onnx.save(model_proto, '06_layernorm.onnx')
print("saved: 06_layernorm.onnx")
