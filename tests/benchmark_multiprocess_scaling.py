import time
import os
import logging
from pathlib import Path
import pandas as pd
from fastembed import TextEmbedding

# Ensure we don't have core contention within processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    input_path = Path("data/texts/sample_data.parquet")
    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()[:1000] 
    
    # Test different parallel process counts
    # We set threads=1 inside each process
    for p_count in [1, 2, 4]:
        logger.info(f"\n--- Testing Parallel Processes: {p_count} ---")
        model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", threads=1)
        
        start_time = time.perf_counter()
        # parallel=p_count uses multiprocessing
        list(model.embed(texts, batch_size=256, parallel=p_count))
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        throughput = len(texts) / duration
        logger.info(f"Processes: {p_count} | Duration: {duration:.2f}s | Throughput: {throughput:.2f} d/s")

if __name__ == "__main__":
    main()
