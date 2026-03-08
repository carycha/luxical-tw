import numpy as np
from luxical_tw.embedder import Embedder

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print(">>> 執行 Luxical 官方 API 語義檢索驗證...")
    
    # 直接載入訓練產出的 npz
    try:
        model = Embedder.load("luxical_one.npz")
        print(f">>> 成功載入模型。詞彙大小: {model.recognized_ngrams.shape[0]} | 維度: {model.embedding_dim}")
    except Exception as e:
        print(f">>> 載入失敗: {e}")
        return

    # 1. 建立測試文案庫
    corpus = [
        "東京迪士尼樂園門票優惠",
        "日本京都溫泉懷石料理",
        "臺北高鐵美食一日遊",
        "高雄捷運城市度假飯店",
        "北海道滑雪假期札幌美食",
        "東京迪士尼五星飯店連泊",
        "台北高鐵美食推薦自由行"
    ]

    # 2. 測試查詢
    queries = ["東京迪士尼", "台北美食"]

    for q_text in queries:
        print(f"\n🔍 查詢句: 【{q_text}】")
        print(f"{'排名':<4} | {'相似度':<8} | {'候選文案內容'}")
        print("-" * 60)

        # 批次產生 Embedding
        q_emb = model([q_text])[0]
        d_embs = model(corpus)
        
        results = []
        for i, doc in enumerate(corpus):
            score = cosine_sim(q_emb, d_embs[i])
            results.append((score, doc))
        
        results.sort(key=lambda x: x[0], reverse=True)

        for i, (score, doc) in enumerate(results[:5]):
            print(f"#{i+1:<3} | {score:10.4f} | {doc}")

    print("\n>>> 驗證完畢。Luxical 官方推理路徑已完全暢通。")

if __name__ == "__main__":
    main()
