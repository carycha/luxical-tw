# Tech Stack: Luxical-TW

## 核心引擎
- **Rust (arrow-tokenize)**：高效能權杖化底層。
- **Python (luxical_tw)**：核心邏輯與訓練框架。
- **Numba**：高效能數值運算 JIT 編譯。

## 中文處理
- **jieba-fast-dat**：基於雙數組 Trie 樹（DAT）的高速分詞引擎。
- **OpenCC (opencc-python-reimplemented)**：簡繁轉換與正規化。
- **PyArrow & Parquet**：高效能資料存儲格式。

## 深度學習
- **PyTorch**：用於對比蒸餾（Contrastive Distillation）訓練。
- **Transformers (Teacher Model)**：調用 `BAAI/bge-m3` 等 SOTA 模型作為蒸餾老師。
