import logging
from typing import Dict, List, Optional

import pyarrow as pa
import jieba_fast_dat
from tqdm.auto import tqdm

from luxical.chinese_utils import ChineseNormalizer

logger = logging.getLogger(__name__)

class ChineseLexicalTokenizer:
    """
    A high-performance Chinese lexical tokenizer using jieba_fast_dat (DAT-based).
    This implementation handles normalization, segmentation, and ID mapping in Python.
    """

    def __init__(self, vocab: Dict[str, int], normalization_config: str = 's2twp'):
        """
        Initializes the tokenizer.
        
        Args:
            vocab: A dictionary mapping words to u32 IDs.
            normalization_config: OpenCC configuration for normalization.
        """
        self.vocab = vocab
        self.normalizer = ChineseNormalizer(config=normalization_config)
        
        # Ensure jieba_fast_dat is initialized
        if not jieba_fast_dat.dt.initialized:
            jieba_fast_dat.initialize()

    def add_words(self, words: List[str]):
        """Adds custom words to the jieba dictionary for better segmentation."""
        for word in words:
            jieba_fast_dat.add_word(word)

    def tokenize_batch(self, texts: pa.Array, batch_size: int = 4096, progress_bar: bool = True) -> pa.ChunkedArray:
        """
        Tokenizes a batch of texts using jieba_fast_dat and Python dict mapping.
        
        Args:
            texts: A pyarrow.StringArray of input texts.
            batch_size: Batch size for processing.
            progress_bar: Whether to show a progress bar.
        Returns:
            A pyarrow.ChunkedArray of token IDs.
        """
        token_chunks = []
        
        # Pre-fetch vocab reference for speed
        vocab = self.vocab
        lcut = jieba_fast_dat.lcut
        normalize = self.normalizer.normalize
        
        with tqdm(total=len(texts), desc="Chinese Tokenizing", disable=not progress_bar) as pbar:
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                batch = texts[start:end]
                
                batch_ids = []
                for t in batch:
                    if not t.is_valid:
                        batch_ids.append(None)
                        continue
                    
                    # 1. Normalize & Segment
                    words = lcut(normalize(t.as_py()))
                    
                    # 2. Map to IDs (Python dict lookup is O(1))
                    ids = [vocab[w] for w in words if w in vocab]
                    batch_ids.append(ids)
                
                # Convert to pyarrow LargeListArray<u32>
                token_pa = pa.array(batch_ids, type=pa.large_list(pa.uint32()))
                token_chunks.append(token_pa)
                
                pbar.update(len(batch))
                
        return pa.chunked_array(token_chunks)
