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
- **多語言與繁體中文支援：** 整合 `jieba_fast_dat` (Double-Array Trie) 與 `OpenCC`，專為通用繁體中文場景優化，具備 100% 的異體字（如台 vs 臺）一致性處理能力。
- **高層次封裝 API (High-level API)：** 提供簡單易用的 `Model` 與 `Trainer` 進入點，將複雜的 Tokenization、BoW 轉換與訓練流程封裝。
- **語義增強與多維度提取：** 支持將長篇文本進行多層次解析與語義增強，將非結構化數據轉化為高品質搜尋特徵。
- **快速 Tokenization：** 使用基於 Rust 的擴展 (`arrow-tokenize`) 與高效 Python 批次處理進行 Tokenization。
- **穩定訓練流程：** 提供針對中文語料優化的密集矩陣 (Dense) 訓練模式，避開 PyTorch 稀疏算子不穩定性。
- **詞彙嵌入 (Lexical Embeddings)：** 生成基於 Token 計數和 TF-IDF 權重的稀疏嵌入。
- **稀疏到稠密 MLP：** 使用緊湊的神經網絡層將稀疏表示投影到稠密嵌入空間。
- **HuggingFace 整合：** 通過 `transformers` 庫提供無縫的推理支持。

## 目標 (Goals)
1. 在保持極高推理速度的同時，實現具備競爭力的嵌入品質。
2. 為訓練和推理提供穩定且易於使用的 Python API。
3. 確保跨平台兼容性（Linux 和 MacOS），並支援最新的 Python 版本。
