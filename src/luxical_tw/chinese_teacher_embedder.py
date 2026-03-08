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

from luxical_tw.misc_utils import fast_8bit_uniform_scalar_quantize

NDArrayOfFloat = NDArray[np.floating[Any]]
NDArrayOfUint8 = NDArray[np.uint8]
logger = logging.getLogger(__name__)

class EmbedderBGEM3:
    """
    Teacher embedder using BGE models for high-quality embeddings.
    """
    HF_MODEL_ID: str = "BAAI/bge-m3"
    QUERY_PREFIX: str = "" 
    DOC_PREFIX: str = ""

    def __init__(self, model_id: str = "BAAI/bge-m3", max_seq_len: int = 512):
        self.model_id = model_id
        logger.info(f"Loading Teacher Model from HuggingFace: `{self.model_id}`")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.device: str | torch.device = "cpu"
        self.max_seq_len = max_seq_len
        
        # 動態偵測輸出維度
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Detected Teacher Embedding Dimension: {self.embedding_dim}")

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
        # BGE models usually use the CLS token (index 0)
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
            embeddings = np.zeros((n, self.embedding_dim), dtype=np.uint8)
        else:
            embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)
            
        for i in tqdm(range(0, n, batch_size), desc="Teacher Embedding", disable=not progress_bar):
            batch_texts = texts[i : i + batch_size]
            inputs = self._tokenize(batch_texts, self.max_seq_len)
            batch_vecs = self._embed_batch(inputs, scalar_quantize_with_limit)
            embeddings[i : i + len(batch_vecs)] = batch_vecs
            
        return embeddings
