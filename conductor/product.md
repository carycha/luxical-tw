# Product Definition: Luxical-TW

Luxical-TW 是高效能文本嵌入框架 Luxical 的繁體中文專用分支。它致力於解決中文語境下 Embedding 速度慢、分詞邏輯不對齊以及對自定義術語支援不足的問題。

## 核心價值
1. **極速 (Speed)**：在 Intel CPU 上提供 SOTA 級別的毫秒級推論。
2. **精準 (Precision)**：原生整合 `jieba-fast-dat` 與專屬繁體字典，解決「台北美食」等專有名詞的分詞斷裂問題。
3. **易用 (Simplicity)**：單一 `.npz` 模型封裝，無須複雜的部署環境。

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
