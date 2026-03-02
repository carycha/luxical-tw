# Initial Concept
Fast lexical dense embeddings built on token counts, TF–IDF, and compact sparse‑to‑dense MLPs.

# Product Definition
## 願景 (Vision)
Luxical 旨在通過結合傳統詞彙方法（如 TF-IDF、Token 計數）與現代神經網絡（MLP），提供極速的文本嵌入解決方案。它填補了簡單快速的詞彙搜索與複雜且耗費計算資源的稠密嵌入之間的空白。

## 目標受眾 (Target Audience)
- 從事大規模文本檢索或分類任務的數據科學家和機器學習工程師。
- 需要在實時應用中使用快速、低延遲文本嵌入的開發者。
- 探索稀疏與稠密文本表示交叉領域的研究人員。

## 核心功能 (Core Features)
- **快速 Tokenization：** 使用基於 Rust 的擴展 (`arrow-tokenize`) 進行高效的 Tokenization。
- **詞彙嵌入 (Lexical Embeddings)：** 生成基於 Token 計數和 TF-IDF 權重的稀疏嵌入。
- **稀疏到稠密 MLP：** 使用緊湊的神經網絡層將稀疏表示投影到稠密嵌入空間。
- **可重現性：** 提供用於重現實驗結果和訓練流程的工具及腳本。
- **HuggingFace 整合：** 通過 `transformers` 庫提供無縫的推理支持。

## 目標 (Goals)
1. 在保持極高推理速度的同時，實現具備競爭力的嵌入品質。
2. 為訓練和推理提供穩定且易於使用的 Python API。
3. 確保跨平台兼容性（Linux 和 MacOS），並支援最新的 Python 版本。
