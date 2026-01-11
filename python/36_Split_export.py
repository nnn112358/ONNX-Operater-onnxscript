# ONNXライブラリをインポート
# Import ONNX libraries
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# Split演算: テンソルの分割
# Split operation: Splitting tensor
# Split演算は複数出力を持つため、onnx.helperを使用
split_data = np.array([2, 2], dtype=np.int64)

# 入力テンソルの定義
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 32, 32])

# 出力テンソルの定義（2つの出力）
output1_tensor = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 2, 32, 32])
output2_tensor = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 2, 32, 32])

# Split初期化子
split_init = helper.make_tensor('split', TensorProto.INT64, [2], split_data)

# Splitノードを定義
node = helper.make_node('Split', inputs=['input', 'split'], outputs=['output1', 'output2'], axis=1)

# ONNXグラフを作成
graph_def = helper.make_graph(
    [node],
    'split_model',
    [input_tensor],
    [output1_tensor, output2_tensor],
    [split_init]
)

# ONNXモデルを作成
model_def = helper.make_model(graph_def, producer_name='onnx-helper')
model_def.opset_import[0].version = 18

# ONNXモデルをファイルに保存
onnx.save(model_def, '36_split.onnx')
print("saved: 36_split.onnx")
