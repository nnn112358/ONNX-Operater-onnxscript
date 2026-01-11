# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# Gemm演算: 一般化行列積
# Gemm operation: General Matrix Multiplication
weight_data = np.random.randn(16, 8).astype(np.float32)
bias_data = np.random.randn(16).astype(np.float32)

@script()
def gemm_model(input: FLOAT[4, 8], weight: FLOAT[16, 8], bias: FLOAT[16]) -> FLOAT[4, 16]:
    return op.Gemm(input, weight, bias, transB=1)

model_proto = gemm_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(weight_data, name='weight'))
model_proto.graph.initializer.append(numpy_helper.from_array(bias_data, name='bias'))
onnx.save(model_proto, '41_gemm.onnx')
print("saved: 41_gemm.onnx")
