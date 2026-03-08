# Luxical-TW — Lexical Dense Embeddings (繁體中文 Fork 版)

![logo](./Luxical_logo.png)

Luxical-TW 是一個專為**大規模繁體中文文本**設計的極速文本嵌入（Embedding）框架。它是原始 Luxical 框架的繁體中文優化版本，結合了詞彙特徵與輕量級神經網路，並原生整合了 **SOTA 級別的中文分詞、正規化與自定義字典支援**。

---

## 🚀 核心優勢 (SOTA Features)

- **原生繁體支援**：內建 OpenCC 正規化與基於 `jieba-fast-dat` 的高速分詞引擎。
- **自動字典載入**：自動讀取 `data/examples/user_dict.txt`，分詞邏輯與訓練完全對齊。
- **極致推理效能**：專為 CPU 優化，單核推理速度可達數千 TPS。
- **自包含模型**：所有權重、詞彙表與分詞器狀態均封裝在單一 `.npz` 檔案中。

---

## 🧠 高層次使用流程 (Standard Workflow)

### 1. 資料預處理 (Data Preprocessing)
從原始 NDJSON 檔案擷取欄位，直接產出 Luxical-TW 訓練所需的 Parquet 格式。

```bash
uv run scripts/ops/extract_ndjson_fields.py \
  --source data/raw.ndjson \
  --dest data/corpus.parquet \
  --fields title
```

### 2. 執行真實訓練 (Real Training)
使用對比蒸餾（Contrastive Distillation）技術進行增量訓練。

```bash
uv run scripts/ops/train_really.py \
  --input data/corpus.parquet \
  --output data/luxical_default.npz \
  --epochs 50
```

### 3. 模型推論 (Inference)
載入模型並產生向量，分詞與正規化會全自動執行。

```python
from luxical_tw.embedder import Embedder

# 載入模型 (自動處理中文分詞與字典)
model = Embedder.load("data/luxical_default.npz")

# 產生向量
embeddings = model(["台北必吃美食推薦", "日本旅遊攻略"])
```

---

## 🛠️ 開發與維護
- **專案名稱**：`luxical-tw`
- **模組名稱**：`luxical_tw`
- **驗證工具**：`uv run scripts/ops/verify_model.py`

詳細技術細節請參閱 `conductor/` 目錄下的文件。
