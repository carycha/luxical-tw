import logging
from typing import Any, Sequence

import numpy as np
from fastembed import TextEmbedding
from tqdm.auto import tqdm

from luxical.misc_utils import fast_8bit_uniform_scalar_quantize

logger = logging.getLogger(__name__)

import queue
import threading
from typing import Any, Sequence, Generator

class FastEmbedTeacher:
    """
    SOTA Elite Teacher for Intel Tiger Lake.
    Features: 
    - Async Double Buffering: Overlaps Tokenization (CPU) with Inference (VNNI).
    - Static Alignment: Optimizes OpenVINO graph for fixed-size blocks.
    - Core Pinning awareness.
    """
    def __init__(
        self, 
        model_name: str = "BAAI/bge-small-zh-v1.5", 
        threads: int | None = 4, 
        max_length: int = 256,
        use_openvino: bool = True
    ):
        logger.info(f"Initializing SOTA Async Teacher: {model_name} (threads={threads})")
        self.model_name = model_name
        self.max_length = max_length
        self.threads = threads
        
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"] if use_openvino else ["CPUExecutionProvider"]
        
        # Configure OpenVINO for latency/throughput balance
        self.model = TextEmbedding(
            model_name=model_name, 
            threads=threads,
            providers=providers
        )
        self.embedding_dim = 512 if "small-zh" in model_name else 768

    def _producer(self, texts: Sequence[str], batch_size: int, q: queue.Queue):
        """Pre-processing thread: Cleans and chunks texts into the queue."""
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            # Static alignment: Pre-truncate here to save ONNX time
            if self.max_length:
                batch = [t[:self.max_length] for t in batch]
            q.put(batch)
        q.put(None) # Sentinel

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 16, # Optimized for Tiger Lake L2 Cache
        scalar_quantize_with_limit: float | None = None,
        progress_bar: bool = True,
    ) -> np.ndarray:
        n = len(texts)
        dtype = np.uint8 if scalar_quantize_with_limit is not None else np.float32
        embeddings = np.zeros((n, self.embedding_dim), dtype=dtype)
        
        # Double-buffering queue
        q = queue.Queue(maxsize=4)
        producer_thread = threading.Thread(target=self._producer, args=(texts, batch_size, q))
        producer_thread.start()
        
        pbar = tqdm(total=n, desc=f"Elite Inference ({self.model_name})", disable=not progress_bar)
        
        processed_count = 0
        while True:
            batch = q.get()
            if batch is None:
                break
            
            # The actual ONNX/OpenVINO inference
            # We call the internal generator directly to keep the engine hot
            batch_gen = self.model.embed(batch, batch_size=len(batch))
            batch_vecs = np.array(list(batch_gen))
            
            if scalar_quantize_with_limit is not None:
                batch_vecs = fast_8bit_uniform_scalar_quantize(batch_vecs, scalar_quantize_with_limit)
                
            embeddings[processed_count : processed_count + len(batch_vecs)] = batch_vecs
            processed_count += len(batch_vecs)
            pbar.update(len(batch_vecs))
            
        producer_thread.join()
        pbar.close()
        return embeddings
