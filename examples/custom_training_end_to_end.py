import json
import numpy as np
from pathlib import Path
from luxical_tw.embedder import Embedder

def main():
    print(">>> Luxical: Custom Model Assembly & Search Example")
    
    # 1. 模擬訓練產出的組件 (實務上這些會從 your_weights.npz 與 vocab.json 載入)
    # 這裡我們建立一個只有 3 個詞的微型模型作為展示
    vocab = {"東京": 0, "台北": 1, "迪士尼": 2, "[UNK]": 3}
    
    # 模擬 2 層 MLP 權重 (輸入維度 4 -> 隱藏層 8 -> 輸出維度 16)
    layers = [
        np.random.randn(4, 8).astype(np.float32),
        np.random.randn(8, 16).astype(np.float32)
    ]
    
    # 2. 使用新 API 一鍵組裝模型
    # 這會自動補全 [CLS], [SEP] 等特殊 Token
    print(">>> 正在使用 Embedder.from_components() 組裝模型...")
    # 預期詞彙大小為 8 (原本 4 + 補全 4)
    # 第一層：(隱藏層 32, 輸入 8)
    # 第二層：(輸出 16, 輸入 32)
    mock_layers = [
        np.random.randn(32, 8).astype(np.float32),
        np.random.randn(16, 32).astype(np.float32)
    ]
    
    model = Embedder.from_components(
        vocab=vocab,
        layers=mock_layers,
        unk_token="[UNK]"
    )
    
    # 3. 儲存模型
    model_path = "example_custom_model.npz"
    model.save(model_path)
    print(f">>> 模型已儲存至: {model_path}")
    
    # 4. 重新載入並測試推理
    print(">>> 正在驗證載入與語義推理...")
    loaded_model = Embedder.load(model_path)
    
    # Debug: Check internal dimensions
    print(f"DEBUG: Layers[0] shape: {loaded_model.bow_to_dense_embedder.layers[0].shape}")
    print(f"DEBUG: Ngram hash map size: {len(loaded_model.ngram_hash_to_ngram_idx)}")
    
    texts = ["我想去東京迪士尼", "台北美食一日遊"]
    embeddings = loaded_model(texts)
    
    print(f"成功為 {len(texts)} 段文字產生向量！")
    print(f"向量維度: {loaded_model.embedding_dim}")
    print("-" * 50)
    print("完成！現在您已掌握如何從自定義組件構建 Luxical 模型。")

if __name__ == "__main__":
    main()
