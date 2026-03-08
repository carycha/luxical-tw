import torch
import torch.nn as nn
import numpy as np
import argparse
import re
import json
import logging
from pathlib import Path
from tqdm import tqdm
from luxical_tw.embedder import Embedder
from luxical_tw.training import contrastive_distillation_loss, equal_beta_adamw
from luxical_tw.chinese_teacher_embedder import EmbedderBGEM3
from luxical_tw.chinese_tokenization import ChineseLexicalTokenizer
from luxical_tw.chinese_utils import ChineseNormalizer
import jieba_fast_dat

# 設定日誌
def setup_logging():
    log_dir = Path("temp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_history.log"
    
    # 清空舊的 log
    if log_file.exists():
        log_file.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):
        h = self.layer1(x)
        out = self.layer2(h)
        return torch.nn.functional.normalize(out, dim=-1)

def train_really(input_text_path: Path, output_model_path: Path, teacher_id: str, epochs: int = 5, lr: float = 2e-3, batch_size: int = 1024, max_samples: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")

    # 1. 載入文字
    if input_text_path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        logger.info(f"正在從 Parquet 載入資料: {input_text_path}")
        table = pq.read_table(input_text_path)
        texts = table["text"].to_pylist()
    else:
        logger.info(f"正在從純文字載入資料: {input_text_path}")
        with open(input_text_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    if max_samples:
        logger.info(f"限制處理資料量為: {max_samples} 筆")
        texts = texts[:max_samples]

    # 2. 建立分詞器與計算詞頻
    logger.info("正在分析語料並建立詞彙表...")
    jieba_fast_dat.initialize()
    user_dict = Path("data/examples/user_dict.txt")
    if user_dict.exists():
        jieba_fast_dat.load_userdict(str(user_dict))

    normalizer = ChineseNormalizer()
    all_segmented_docs = []
    token_counts = {}
    
    for t in tqdm(texts, desc="分詞與詞頻統計"):
        words = [w for w in jieba_fast_dat.lcut(normalizer.normalize(t)) if len(w) >= 2]
        all_segmented_docs.append(words)
        for w in words:
            token_counts[w] = token_counts.get(w, 0) + 1
    
    # 建立詞彙表
    vocab = {word: i for i, word in enumerate(sorted(token_counts.keys()))}
    vocab["[UNK]"] = len(vocab)
    
    # 詞彙統計檢查
    total_unique_words = len(token_counts)
    logger.info(f"--- 詞彙量統計 ---")
    logger.info(f"從訓練集中提取的原始詞彙量: {total_unique_words:,d}")
    logger.info(f"最終詞彙表大小 (含 [UNK]): {len(vocab):,d}")
    
    # 檢查詞彙覆蓋 (在此案例中應為 100%，因為是從 texts 提取)
    used_in_training = sum(1 for w in vocab if w in token_counts)
    coverage = (used_in_training / total_unique_words) * 100 if total_unique_words > 0 else 0
    logger.info(f"訓練詞彙覆蓋率: {coverage:.2f}% ({used_in_training:,d}/{total_unique_words:,d})")
    
    full_vocab = vocab.copy()
    for token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]:
        if token not in full_vocab:
            full_vocab[token] = len(full_vocab)
            
    input_dim = len(full_vocab)
    logger.info(f"神經網路輸入維度 (含特殊 tokens): {input_dim:,d}")

    # 計算 IDF
    total_tokens = sum(token_counts.values())
    idf_values = np.ones(input_dim, dtype=np.float32)
    for word, idx in vocab.items():
        if word in token_counts:
            idf_values[idx] = np.log(total_tokens / token_counts[word])
    
    avg_idf = np.mean(idf_values[:len(vocab)-1]) if len(vocab) > 1 else 1.0
    idf_values[len(vocab):] = avg_idf

    # 3. 獲取 Teacher Embeddings
    logger.info(f"正在獲取 Teacher ({teacher_id}) 的目標向量...")
    teacher = EmbedderBGEM3(model_id=teacher_id)
    teacher.to(device)
    output_dim = teacher.embedding_dim
    hidden_dim = 128
    target_embeddings = teacher.embed_texts(texts, batch_size=32)

    # 4. 初始化 Student
    student = StudentModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = equal_beta_adamw(student.parameters(), lr=lr)

    # 5. 訓練迴圈
    logger.info(f"開始訓練 ({epochs} Epochs, Batch Size: {batch_size})...")
    num_samples = len(texts)
    
    for epoch in range(epochs):
        epoch_loss = 0
        student.train()
        indices = np.random.permutation(num_samples)
        
        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i : i + batch_size]
            batch_docs = [all_segmented_docs[idx] for idx in batch_idx]
            batch_targets = torch.from_numpy(target_embeddings[batch_idx]).to(device)
            
            batch_tfidf = []
            for doc in batch_docs:
                vec = np.zeros(input_dim, dtype=np.float32)
                for w in doc:
                    if w in full_vocab:
                        vec[full_vocab[w]] += 1.0
                vec = vec * idf_values
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                batch_tfidf.append(vec)
            
            tfidf_tensor = torch.from_numpy(np.array(batch_tfidf)).to(device)
            
            optimizer.zero_grad()
            student_out = student(tfidf_tensor)
            loss = contrastive_distillation_loss(
                student_embedding_batch=student_out,
                teacher_embedding_batch=batch_targets,
                loss_temperature=0.05
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / (num_samples / batch_size)
        logger.info(f"Epoch {epoch+1}/{epochs} 完成 - 平均 Loss: {avg_loss:.6f}")

    # 6. 儲存模型
    logger.info("訓練完成，正在儲存模型...")
    trained_layers = [
        student.layer1.weight.detach().cpu().numpy(),
        student.layer2.weight.detach().cpu().numpy()
    ]
    
    final_model = Embedder.from_components(
        vocab=vocab, 
        layers=trained_layers, 
        idf_values=idf_values,
        unk_token="[UNK]"
    )
    final_model.save(output_model_path)
    logger.info(f"模型已儲存至: {output_model_path}")
    logger.info(f"詳細訓練日誌請見: temp/train_history.log")

def main():
    parser = argparse.ArgumentParser(description="執行 Luxical-TW 訓練並記錄日誌。")
    parser.add_argument("--input", type=str, default="data/examples/corpus_sample.parquet")
    parser.add_argument("--output", type=str, default="data/luxical_default.npz")
    parser.add_argument("--teacher", type=str, default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    train_really(Path(args.input), Path(args.output), args.teacher, args.epochs, args.lr, args.batch_size, args.max_samples)

if __name__ == "__main__":
    main()
