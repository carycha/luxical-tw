import json
import numpy as np
from scipy.sparse import csr_matrix
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def text_to_bow(text, vocab, dim):
    indices = []
    data = []
    # 這裡模擬最基礎的空格分詞，測試時請確保詞彙在 vocab.json 中
    tokens = text.split()
    for t in tokens:
        if t in vocab:
            indices.append(vocab[t])
            data.append(1)
    
    sorted_pairs = sorted(zip(indices, data))
    if not sorted_pairs:
        return csr_matrix((1, dim), dtype=np.float32)
    
    indices, data = zip(*sorted_pairs)
    return csr_matrix((data, indices, [0, len(indices)]), shape=(1, dim), dtype=np.float32)

def main():
    # 1. 載入模型組件
    with open('models/custom_model/vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_dim = len(vocab)
    
    weights_data = np.load('models/custom_model/weights.npz')
    layers = [weights_data[f] for f in weights_data.files]
    mlp = SparseToDenseEmbedder(layers=layers)

    def get_embedding(text):
        bow = text_to_bow(text, vocab, vocab_dim)
        if bow.nnz == 0: return None
        emb = np.empty((1, mlp.output_dim), dtype=np.float32)
        mlp(bow, out=emb)
        return emb[0]

    # 2. 建立候選文案庫 (Corpus) - 故意混合不同城市與主題
    corpus = [
        "東京 迪士尼 樂園 門票 優惠",
        "日本 京都 溫泉 懷石 料理 體驗",
        "臺北 高鐵 美食 一日 遊 推薦",
        "高雄 捷運 城市 度假 飯店 入住",
        "首爾 時尚 購物 之 旅 推薦",
        "北海道 滑雪 假期 札幌 美食 饗宴",
        "東京 迪士尼 連泊 五星 飯店",
        "台北 高鐵 美食 推薦 自由 行"
    ]

    # 3. 執行搜尋測試
    # 我們測試兩個不同方向的查詢
    queries = ["東京 迪士尼", "台北 美食"]

    for q_text in queries:
        q_emb = get_embedding(q_text)
        if q_emb is None: continue

        print(f"\n🔍 查詢句: 【{q_text}】")
        print(f"{'排名':<4} | {'相似度':<8} | {'候選文案內容'}")
        print("-" * 60)

        # 計算所有候選句的分數
        scored_results = []
        for doc in corpus:
            d_emb = get_embedding(doc)
            if d_emb is not None:
                score = cosine_sim(q_emb, d_emb)
                scored_results.append((score, doc))
        
        # 由高到低排序
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # 輸出前 5 名
        for i, (score, doc) in enumerate(scored_results[:5]):
            status = "✅ (正確命中)" if i == 0 else ""
            print(f"#{i+1:<3} | {score:10.4f} | {doc} {status}")

    print("\n💡 觀察重點：")
    print("1. 當查詢『東京 迪士尼』時，前兩名是否都是東京相關的產品？")
    print("2. 當查詢『台北 美食』時，模型是否能正確抓到『臺北』(異體字) 且把台北相關文案排在最前面？")

if __name__ == "__main__":
    main()
