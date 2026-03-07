# Luxical — Lexical Dense Embeddings (繁體中文版)

![logo](./Luxical_logo.png)

Luxical 是一個結合了詞彙特徵（Token Counts, TF-IDF）與輕量級神經網路（Sparse-to-Dense MLP）的極速文本嵌入（Embedding）框架。它能在保持極高推理速度的同時，透過教師模型（Teacher Model）的蒸餾實現高品質的語義表示。

本分支出色地優化了 **繁體中文支援** 並針對 **Intel CPU (Tiger Lake+)** 實現了 SOTA 級別的加速。

---

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境
本專案使用 `uv` 管理。您需要安裝 Rust 工具鏈來編譯高效能的斷詞引擎。

```bash
# 安裝 Rust (若尚未安裝)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 克隆與同步
git clone https://github.com/DatologyAI/luxical-tw.git
cd luxical-tw
uv sync

# 編譯 Rust 斷詞模組
cd arrow_tokenize && uv run maturin develop --release && cd ..
```

---

## 🧠 高層次 API 使用 (High-level API)

Luxical 提供簡單直觀的 `Model` 與 `Trainer` 介面，適用於任何中文文本領域。

### 1. 模型推論 (Inference)
載入模型並產生向量：

```python
import luxical
import numpy as np

# 1. 載入模型
model = luxical.Model.load("models/my_model.npz")

# 2. 定義測試句子
sentences = ["這是一個關於語義相似度的測試", "我們正在驗證文本嵌入的效果"]

# 3. 產生 Embedding (Tokenize -> BoW -> TF-IDF -> MLP)
embeddings = model.encode(sentences)

# 4. 計算餘弦相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score = cosine_sim(embeddings[0], embeddings[1])
print(f"相似度分數: {score:.4f}")
```

### 2. 模型訓練 (Training)
使用 `Trainer` 封裝整個蒸餾訓練流程：

```python
from luxical import Trainer

# 1. 從 Ngram Summary 初始化 Trainer
trainer = Trainer.from_ngram_summary(
    ngram_summary_path="data/ngram_summary.npz",
    tokenizer_id="google-bert/bert-base-uncased",
    batch_size=8192,
    device="cpu"
)

# 2. 開始訓練 (串流處理文字語料與教師向量)
trainer.train(
    text_dataset_path="data/texts/",
    teacher_emb_dataset_path="data/embeddings/",
    num_epochs=3.0,
    teacher_emb_quantization_limit=1.0
)

# 3. 儲存模型
trainer.save("models/new_luxical_model.npz")
```

---

## 📂 目錄結構 (Project Layout)

*   `src/luxical/`：通用核心庫 (Model, Trainer, Tokenization, Chinese Utils)。
*   `projects/`：(已忽略) 存放使用者自定義的訓練專案、特定領域數據與私有模型。
*   `examples/`：通用範例與 Demo 腳本。
*   `tools/`：基準測試 (Benchmark) 與效能驗證工具。
*   `data/`：存放全域性的原始數據。
*   `models/`：存放全域性的模型權重。
*   `tests/`：單元測試與整合測試。

---

## 🛠️ 開發與貢獻
*   **測試**：`uv run pytest tests/`
*   **類型檢查**：`uv run pyright`
*   **代碼風格**：`uv run ruff check .`

更多詳細資訊請參閱 [DatologyAI 部落格](https://www.datologyai.com/blog/introducing-luxical-embeddings)。
