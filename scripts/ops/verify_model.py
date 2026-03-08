import numpy as np
import time
import argparse
from pathlib import Path
from luxical_tw.embedder import Embedder
import pyarrow.parquet as pq

# Luxical 全面調整為預設使用中文分詞器。
# 本驗證腳本現在直接使用官方 API，不需手動處理分詞。

def cosine_similarity(a, b):
    # a: (1, D), b: (N, D)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)

def verify_model(model_path: Path, parquet_path: Path, num_samples: int = 5000):
    print(f"\n🚀 [Luxical 深度驗證] 正在啟動引擎 (預設使用 SOTA 中文分詞)...")
    
    # 1. 載入模型
    if not model_path.exists():
        print(f"❌ 錯誤: 找不到模型檔案 {model_path}")
        return
    
    # Embedder 現在會自動處理中文分詞與字典
    model = Embedder.load(model_path)
    print(f"✅ 模型載入完成 | 維度: {model.embedding_dim}")

    # 2. 載入真實語料庫
    if not parquet_path.exists():
        print(f"❌ 錯誤: 找不到語料檔案 {parquet_path}")
        return
    
    print(f"📂 正在從 {parquet_path.name} 讀取前 {num_samples} 筆標題作為檢索庫...")
    table = pq.read_table(parquet_path)
    corpus = table["text"].to_pylist()[:num_samples]
    
    print("⚡ 正在為檢索庫產生向量 (全面自動化分詞)...")
    start_time = time.time()
    corpus_embeddings = model(corpus) # 直接呼叫官方 API
    duration = time.time() - start_time
    print(f"✅ 向量化完成 | 速度: {len(corpus) / duration:.1f} TPS")

    # 3. 測試查詢
    test_queries = [
        "台北必吃拉麵推薦",
        "日本東京迪士尼攻略",
        "化妝品開箱分享",
        "全家便利商店新品試吃",
        "台中兩天一夜旅遊行程"
    ]

    print("\n" + "="*80)
    print(f"{'語義檢索 Top 5 測試結果 (全自動模式)':^80}")
    print("="*80)

    query_embeddings = model(test_queries)

    for i, q_text in enumerate(test_queries):
        print(f"\n🔍 查詢句: 【{q_text}】")
        
        q_vec = query_embeddings[i : i+1]
        similarities = cosine_similarity(q_vec, corpus_embeddings)[0]
        
        # 取得 Top 5 索引
        top_indices = np.argsort(similarities)[::-1][:5]
        
        print(f"{'排名':<4} | {'相似度':<8} | {'標題內容'}")
        print("-" * 80)
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            content = corpus[idx]
            print(f"#{rank+1:<3} | {score:.4f} | {content}")

    print("\n" + "="*80)
    print("💡 提示: 相似度具備階層區分且內容相關，代表分詞引擎已在 Embedder 內部完美運作。")
    print(">>> 驗證完畢。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/luxical_default.npz")
    parser.add_argument("--parquet", type=str, default="data/examples/corpus_sample.parquet")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    
    verify_model(Path(args.model), Path(args.parquet), args.samples)
