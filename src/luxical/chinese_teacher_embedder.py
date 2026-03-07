import logging
from multiprocessing.pool import ThreadPool
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from luxical.misc_utils import fast_8bit_uniform_scalar_quantize

NDArrayOfFloat = NDArray[np.floating[Any]]
NDArrayOfUint8 = NDArray[np.uint8]
logger = logging.getLogger(__name__)

class EmbedderBGEM3:
    """
    Teacher embedder using BGE-M3 for high-quality multi-lingual (Traditional Chinese) embeddings.
    """
    HF_MODEL_ID: str = "BAAI/bge-m3"
    QUERY_PREFIX: str = "" # BGE-M3 usually doesn't need explicit prefixes unless specified
    DOC_PREFIX: str = ""
    EMBEDDING_DIM: int = 1024

    def __init__(self, max_seq_len: int = 512):
        logger.info(f"Loading BGE-M3 from HuggingFace: `{self.HF_MODEL_ID}`")
        self.tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        self.model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self.device: str | torch.device = "cpu"
        self.max_seq_len = max_seq_len

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> None:
        self.device = device
        self.model = self.model.to(device, dtype=dtype)
        self.model.eval()

    def _tokenize(self, texts: Sequence[str], max_seq_len: int | None) -> dict[str, Tensor]:
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def _embed_batch(
        self, 
        inputs: dict[str, Tensor],
        scalar_quantize_with_limit: float | None = None
    ) -> NDArrayOfFloat | NDArrayOfUint8:
        outputs = self.model(**inputs)
        # BGE-M3 usually uses the CLS token (index 0)
        vec = outputs.last_hidden_state[:, 0]
        vec = F.normalize(vec, dim=-1)
        vec = vec.float().cpu().numpy()
        
        if scalar_quantize_with_limit is not None:
            vec = fast_8bit_uniform_scalar_quantize(vec, scalar_quantize_with_limit)
        return vec

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        scalar_quantize_with_limit: float | None = None,
        progress_bar: bool = True,
    ) -> NDArrayOfFloat | NDArrayOfUint8:
        n = len(texts)
        if scalar_quantize_with_limit is not None:
            embeddings = np.zeros((n, self.EMBEDDING_DIM), dtype=np.uint8)
        else:
            embeddings = np.zeros((n, self.EMBEDDING_DIM), dtype=np.float32)
            
        for i in tqdm(range(0, n, batch_size), desc="Teacher Embedding", disable=not progress_bar):
            batch_texts = texts[i : i + batch_size]
            inputs = self._tokenize(batch_texts, self.max_seq_len)
            batch_vecs = self._embed_batch(inputs, scalar_quantize_with_limit)
            embeddings[i : i + len(batch_vecs)] = batch_vecs
            
        return embeddings
