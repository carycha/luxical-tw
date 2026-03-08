import time
import os
import torch
import numpy as np
import luxical_tw.embedder
from transformers import AutoModel
import torch.nn.functional as F

TEST_DATA = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps above a sleepy hound.",
    "Artificial intelligence is transforming the way we work and live.",
    "Luxical embeddings are designed to be extremely fast and efficient on CPU.",
    "Data science involves extracting insights from complex datasets.",
    "The weather today is sunny and bright, perfect for a walk.",
    "Machine learning models can identify patterns in unstructured data.",
    "Natural language processing helps computers understand human speech.",
    "Python is the leading programming language for data analysis.",
    "Fast text embeddings are crucial for large-scale retrieval tasks."
] * 100

def main():
    print("🚀 --- Luxical 效能與資源基準測試 ---")

    print("\n📦 [1/4] 載入模型...")
    try:
        start_load = time.time()
        model = AutoModel.from_pretrained("datologyai/luxical_tw-one", trust_remote_code=True)
        load_time = time.time() - start_load
        print(f"   - 載入耗時: {load_time:.4f} 秒")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        return
    
    print("\n⚡ [2/4] 測試處理效能 (1000 條句子)...")
    batch_size = 100
    total_sentences = len(TEST_DATA)
    start_time = time.time()
    
    for i in range(0, total_sentences, batch_size):
        batch = TEST_DATA[i:i + batch_size]
        with torch.no_grad():
            _ = model(batch).embeddings
            
    total_time = time.time() - start_time
    throughput = total_sentences / total_time
    print(f"   - 總處理耗時: {total_time:.4f} 秒")
    print(f"   - 吞吐量 (Throughput): {throughput:.2f} 條句子/秒")
    print(f"   - 平均每條耗時: {(total_time / total_sentences)*1000:.4f} 毫秒")

    print("\n📊 [3/4] 資源估計")
    print("   - 模型文件大小: ~930 MB")
    print("   - 預期記憶體佔用: 1.0 - 1.5 GB")
    print("   - 硬體：僅使用 CPU (單執行緒或多執行緒視 PyTorch 配置而定)")

    print("\n🧠 [4/4] 語義驗證測試:")
    s1, s2 = ["The quick fox"], ["A fast canine"]
    v1, v2 = model(s1).embeddings, model(s2).embeddings
    sim = F.cosine_similarity(v1, v2).item()
    print(f"   - 'The quick fox' vs 'A fast canine' 相似度: {sim:.4f}")

if __name__ == "__main__":
    main()
