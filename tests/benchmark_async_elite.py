import time
import os
import logging
from pathlib import Path
import pandas as pd
from luxical_tw.fast_teacher_embedder import FastEmbedTeacher

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load 4000 rows for steady-state testing
    input_path = Path("data/texts/sample_data.parquet")
    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()[:4000] 
    
    logger.info(f"\n>>> Running SOTA Async Benchmark | Size: {len(texts)}")
    
    # Initialize the updated Elite Teacher
    # threads=4 to match physical cores, use_openvino=True for hardware acceleration
    teacher = FastEmbedTeacher(
        model_name="BAAI/bge-small-zh-v1.5", 
        threads=4, 
        max_length=256,
        use_openvino=True
    )
    
    # Warmup
    teacher.embed_texts(["warmup"], batch_size=1, progress_bar=False)
    
    start_time = time.perf_counter()
    # Use batch_size=16 as the sweet spot for Tiger Lake
    embeddings = teacher.embed_texts(
        texts, 
        batch_size=16, 
        scalar_quantize_with_limit=1.0, 
        progress_bar=True
    )
    duration = time.perf_counter() - start_time
    
    throughput = len(texts) / duration
    logger.info(f"\nFinal Result:")
    logger.info(f"Duration: {duration:.2f}s | Throughput: {throughput:.2f} d/s")
    
    # Sanity check on output
    logger.info(f"Embeddings shape: {embeddings.shape} | Dtype: {embeddings.dtype}")

if __name__ == "__main__":
    main()
