import numpy as np
import pytest
<<<<<<< HEAD
from luxical.fast_teacher_embedder import FastEmbedTeacher
=======
from luxical_tw.fast_teacher_embedder import FastEmbedTeacher
>>>>>>> f1cbaea (reinit)

def test_fast_embed_teacher_initialization():
    """Test that the FastEmbedTeacher initializes correctly with the requested model."""
    teacher = FastEmbedTeacher(model_name="BAAI/bge-small-zh-v1.5")
    assert teacher.model_name == "BAAI/bge-small-zh-v1.5"
    assert teacher.embedding_dim == 512

def test_fast_embed_teacher_embedding():
    """Test that embedding generation works and produces normalized float32 vectors."""
    teacher = FastEmbedTeacher(model_name="BAAI/bge-small-zh-v1.5")
    texts = ["你好，這是一個測試。", "Hello, this is a test."]
    embeddings = teacher.embed_texts(texts, batch_size=2)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32
    
    # Check normalization (FastEmbed usually returns normalized vectors by default)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_fast_embed_teacher_dimension_detection():
    """Test that dimension detection fallback works for non-standard models."""
    teacher = FastEmbedTeacher(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert teacher.embedding_dim == 384

def test_fast_embed_teacher_quantization():
    """Test that 8-bit quantization is correctly applied when requested."""
    teacher = FastEmbedTeacher(model_name="BAAI/bge-small-zh-v1.5")
    texts = ["測試量化功能。"]
    limit = 1.0
    embeddings = teacher.embed_texts(texts, batch_size=1, scalar_quantize_with_limit=limit)
    
    assert embeddings.dtype == np.uint8
    assert embeddings.shape == (1, 512)
    assert np.all(embeddings >= 0)
    assert np.all(embeddings <= 255)
