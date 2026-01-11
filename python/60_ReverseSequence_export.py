# ONNXライブラリをインポート
# Import ONNX libraries
import onnxscript
from onnxscript import FLOAT, INT64, BOOL, script
from onnxscript import opset18 as op
import numpy as np
import onnx
from onnx import numpy_helper

# ReverseSequence演算: シーケンスの逆順
# ReverseSequence operation: Reversing sequence
seq_lens = np.array([5, 5, 5, 5], dtype=np.int64)

@script()
def reverse_sequence_model(input: FLOAT[4, 5, 10], sequence_lens: INT64[4]) -> FLOAT[4, 5, 10]:
    return op.ReverseSequence(input, sequence_lens, batch_axis=0, time_axis=1)

model_proto = reverse_sequence_model.to_model_proto()
model_proto.graph.initializer.append(numpy_helper.from_array(seq_lens, name='sequence_lens'))
onnx.save(model_proto, '60_reverse_sequence.onnx')
print("saved: 60_reverse_sequence.onnx")
