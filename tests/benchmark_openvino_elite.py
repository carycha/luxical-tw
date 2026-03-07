import time
import os
import logging
from pathlib import Path
import pandas as pd
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_elite_benchmark(provider: str, threads: int, batch_size: int, texts: list):
    """Run embedding benchmark for a specific SOTA config."""
    logger.info(f"\n>>> Running: {provider} | Threads: {threads} | Batch: {batch_size}")
    
    # TextEmbedding in fastembed handles providers selection
    providers = [provider] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
    
    try:
        model = TextEmbedding(
            model_name="BAAI/bge-small-zh-v1.5", 
            threads=threads,
            providers=providers
        )
        
        # Warmup
        list(model.embed(["warmup"], batch_size=1))
        
        start_time = time.perf_counter()
        list(model.embed(texts, batch_size=batch_size))
        duration = time.perf_counter() - start_time
        
        throughput = len(texts) / duration
        logger.info(f"Duration: {duration:.2f}s | Throughput: {throughput:.2f} d/s")
        return throughput
    except Exception as e:
        logger.error(f"Failed to run config: {e}")
        return 0

def main():
    input_path = Path("data/texts/sample_data.parquet")
    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()[:1000] 
    
    results = []
    # Configurations to compare
    configs = [
        ("CPUExecutionProvider", 4, 256),
        ("OpenVINOExecutionProvider", 4, 256),
        ("OpenVINOExecutionProvider", 4, 32),
        ("OpenVINOExecutionProvider", 4, 16),
    ]
    
    for provider, threads, batch in configs:
        tput = run_elite_benchmark(provider, threads, batch, texts)
        results.append(tput)

if __name__ == "__main__":
    main()
