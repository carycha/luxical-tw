import time
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from fastembed import TextEmbedding

# Reset environment variables to allow multicore experimentation
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)
os.environ.pop("OPENBLAS_NUM_THREADS", None)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_benchmark(threads: int, batch_size: int, texts: list):
    """Run embedding benchmark for a specific config."""
    # TextEmbedding handles onnxruntime session options
    model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", threads=threads)
    
    # Pre-warm to ensure model is loaded and JIT'd
    list(model.embed(["warmup sentence"], batch_size=1))
    
    start_time = time.perf_counter()
    # model.embed returns a generator, consume it fully to measure time
    list(model.embed(texts, batch_size=batch_size))
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    throughput = len(texts) / duration
    return duration, throughput

def main():
    input_path = Path("data/texts/sample_data.parquet")
    if not input_path.exists():
        logger.error(f"Test data not found at {input_path}")
        return
        
    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()[:500] # Use 500 texts for fast iteration
    
    results = []
    configs = [
        (1, 32),
        (1, 128),
        (2, 64),
        (4, 64),
        (4, 256),
    ]
    
    print(f"{'Threads':>8} | {'BatchSize':>9} | {'Duration (s)':>13} | {'Throughput (d/s)':>16}")
    print("-" * 55)
    
    for threads, batch_size in configs:
        duration, throughput = run_benchmark(threads, batch_size, texts)
        print(f"{threads:>8} | {batch_size:>9} | {duration:>13.4f} | {throughput:>16.2f}")
        results.append({"threads": threads, "batch": batch_size, "tput": throughput})

if __name__ == "__main__":
    main()
