# Tech Stack
## 核心語言 (Core Languages)
- **Python (>=3.11)：** 用於 MLP、嵌入邏輯和整個系統的編排。
- **Rust：** 用於 `arrow-tokenize` 軟體包，以提供高效能的 Tokenization。

## 機器學習與效能 (Machine Learning & Performance)
- **PyTorch (>=2.5.0)：** 用於實作與訓練「稀疏到稠密」的 MLP。
- **Numba (>=0.61.2)：** 用於優化的數值運算。
- **SciPy (>=1.15.3)：** 用於稀疏矩陣操作與數值演算法。
- **Scikit-learn (>=1.7.0)：** 選擇性地用於輔助機器學習工具。

## 數據與模型處理 (Data & Model Handling)
- **HuggingFace Transformers：** 用於與預訓練模型的無縫整合與推理。
- **Tqdm：** 為訓練和推理過程提供進度條。
- **S3FS & FSSpec：** 支援文件系統抽象與雲端存儲 (S3)。

## 開發與構建工具 (Development & Build Tools)
- **Hatchling：** Python 軟體包的構建後端。
- **Maturin：** Rust 擴充程式碼的構建工具。
- **Pytest：** 測試框架。
- **Pyright：** 靜態類型檢查。
- **Ruff：** 程式碼檢查與格式化。
- **Uniplot：** 基於 CLI 的繪圖工具，用於快速數據視覺化。

## 基礎架構與環境 (Infrastructure & Environment)
- **單體庫 (Monorepo) 結構：** 使用 `uv` 工作區 (workspace) 管理。
- **作業系統支援：** 針對 Linux 和 MacOS 進行優化。
- **虛擬環境管理：** 通過 `uv` 管理標準虛擬環境。
