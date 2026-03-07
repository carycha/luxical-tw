import time
import logging
import cProfile
import pstats
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

# Force single-threading at the library level
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from luxical.fast_teacher_embedder import FastEmbedTeacher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_task():
    input_path = Path("data/texts/my_data_small.parquet")
    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()
    
    # Initialize teacher with further optimized parameters
    teacher = FastEmbedTeacher(model_name="BAAI/bge-small-zh-v1.5", threads=1, max_length=256)
    
    start_time = time.perf_counter()
    
    # Run embedding with larger batch size
    embeddings = teacher.embed_texts(
        texts,
        batch_size=128,
        scalar_quantize_with_limit=1.0,
        progress_bar=False
    )
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info(f"Total duration for {len(texts)} texts: {duration:.4f}s")
    logger.info(f"Throughput: {len(texts) / duration:.2f} docs/sec")
    
    return duration

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    profile_task()
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
