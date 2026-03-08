import time
import torch
import torch.nn.functional as F
from transformers import AutoModel

def compute_similarity(v1, v2):
    return F.cosine_similarity(v1, v2).item()

print("🚀 正在從 HuggingFace 載入 Luxical-One 模型 (這可能需要 30-60 秒)...")
try:
    # 載入 DatologyAI 預訓練好的模型
    model = AutoModel.from_pretrained("datologyai/luxical_tw-one", trust_remote_code=True)

    # 準備測試句子
    sentences = [
        "The quick brown fox jumps over the lazy dog.",          # 原始句子
        "A fast auburn canine leaps above a sleepy hound.",      # 意思相同，但單詞不同
        "I love eating fresh apples and oranges in the summer."  # 不相關
    ]

    print("\n--- 📝 測試句子 ---")
    for i, s in enumerate(sentences):
        print(f"句子 {i+1}: {s}")

    # 1. 測試速度與嵌入
    print("\n--- ⚡ 執行嵌入 (Embedding) ---")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(sentences)
        embeddings = outputs.embeddings
    end_time = time.time()

    print(f"✅ 完成！處理 {len(sentences)} 個句子耗時: {(end_time - start_time)*1000:.2f} 毫秒")

    # 2. 測試相似度
    sim_1_2 = compute_similarity(embeddings[0:1], embeddings[1:2])
    sim_1_3 = compute_similarity(embeddings[0:1], embeddings[2:3])

    print("\n--- 🧠 語義相似度結果 (Cosine Similarity) ---")
    print(f"句子 1 與 句子 2 (意思相同，單詞不同): {sim_1_2:.4f}")
    print(f"句子 1 與 句子 3 (內容完全不相關): {sim_1_3:.4f}")

    if sim_1_2 > sim_1_3:
        print("\n💡 結果分析：Luxical 成功識別了語義相似度！")
        print("即使句子 2 使用了完全不同的單詞 (auburn, canine, leaps, hound)，")
        print("Luxical 依然能辨識出它與句子 1 的語義非常接近。")
        print("這就是「詞彙稠密嵌入 (Lexical Dense)」的力量。")

except Exception as e:
    print(f"\n❌ 執行失敗: {e}")
    print("請確保您已經執行過 `uv sync` 且網路連線正常。")
