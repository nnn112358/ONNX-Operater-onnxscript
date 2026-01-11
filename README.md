# ONNX Operators Sample Collection

ONNXの主要なオペレータ60種類の実装サンプル集です。onnxscriptを使用して直接ONNXモデルを作成し、推論のコード例を提供しています。

## 対応オペレータ

全60種類（ニューラルネットワーク層、活性化関数、数学演算、テンソル操作、線形代数、比較演算、集約・統計演算など）をカバー。

詳細は [onnx_operators.md](onnx_operators.md) を参照。

各オペレータに以下の2ファイルを用意：
- `XX_OperatorName_export.py` - onnxscriptを使用してONNXモデルを作成
- `XX_OperatorName_inference.py` - ONNXモデルで推論を実行

## セットアップ

### 必要な環境
- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー

### インストール

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# または
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 依存関係のインストール
cd python
uv sync
```

### 依存ライブラリ
- onnxscript
- onnx (>=1.16.0)
- onnxruntime (>=1.18.0)
- numpy

## 使い方

```bash
cd python

# モデル作成（onnxscriptでONNXモデルを作成）
uv run 01_Conv_export.py

```

他のオペレータも同様です（例：`uv run 14_Softmax_export.py`）

### すべてのオペレータを一括実行

01から60までのすべてのオペレータを順番に実行するスクリプトも用意しています。

```bash
cd python
./run_all.sh
```

このスクリプトは各オペレータのエクスポートと推論を順番に実行し、エラーが発生した場合は処理を停止します。

> **Note**: `uv run` は自動的に仮想環境を使用するため、activate不要です。

## ライセンス

MIT License
