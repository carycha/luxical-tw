import json
import numpy as np
import pytest
import jieba_fast_dat
from pathlib import Path
from scipy.sparse import csr_matrix
from luxical.chinese_utils import ChineseNormalizer
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder

class ChineseModelTester:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        weights = np.load(self.model_dir / "weights.npz")
        layers = [weights[f] for f in weights.files]
        self.embedder = SparseToDenseEmbedder(layers=layers)
        self.normalizer = ChineseNormalizer()
        jieba_fast_dat.initialize()

    def get_embedding(self, text: str):
        normalized = self.normalizer.normalize(text)
        words = jieba_fast_dat.lcut(normalized)
        row_data, row_indices = [], []
        for w in words:
            if w in self.vocab:
                row_indices.append(self.vocab[w])
                row_data.append(1.0)
        if not row_data: return None
        q_sparse = csr_matrix((row_data, row_indices, [0, len(row_data)]), shape=(1, len(self.vocab)))
        q_sparse.data = np.sqrt(q_sparse.data).astype(np.float32)
        return self.embedder(q_sparse)[0]

def cosine_sim(a, b):
    if a is None or b is None: return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@pytest.fixture
def tester():
    model_path = "models/techbang_luxical"
    if not Path(model_path).exists():
        pytest.skip("Model not found")
    return ChineseModelTester(model_path)

def test_semantic_similarity(tester):
    query = "這台旗艦手機的拍照功能非常強大"
    pos = "手機相機實測與攝影評價"
    neg = "如何在家做出好吃的日本料理"
    emb_q = tester.get_embedding(query)
    emb_pos = tester.get_embedding(pos)
    emb_neg = tester.get_embedding(neg)
    sim_pos = cosine_sim(emb_q, emb_pos)
    sim_neg = cosine_sim(emb_q, emb_neg)
    print(f"\nSimilarity (Positive): {sim_pos:.4f}")
    print(f"Similarity (Negative): {sim_neg:.4f}")
    assert sim_pos > sim_neg

def test_variant_consistency(tester):
    txt1 = "台北最新的科技趨勢"
    txt2 = "臺北最新的科技趨勢"
    emb1 = tester.get_embedding(txt1)
    emb2 = tester.get_embedding(txt2)
    sim = cosine_sim(emb1, emb2)
    print(f"\nVariant Similarity (台 vs 臺): {sim:.4f}")
    assert sim > 0.99
