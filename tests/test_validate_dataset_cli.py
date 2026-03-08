import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

<<<<<<< HEAD
from luxical.scripts.validate_dataset import (
=======
from luxical_tw.scripts.validate_dataset import (
>>>>>>> f1cbaea (reinit)
    create_parser,
    validate_text_dataset,
    validate_embedding_dataset,
)

@pytest.fixture
def temp_datasets():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        text_dir = tmp_path / "texts"
        emb_dir = tmp_path / "embeddings"
        text_dir.mkdir()
        emb_dir.mkdir()
        
        # Valid text data
        text_table = pa.table({
            "id": ["1", "2", "3"],
            "text": ["hello", "world", "test"]
        })
        pq.write_table(text_table, text_dir / "data.parquet")
        
        # Valid embedding data
        emb_table = pa.table({
            "document_id": ["1", "2", "3"],
            "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        })
        pq.write_table(emb_table, emb_dir / "data.parquet")
        
        yield text_dir, emb_dir

def test_validate_text_dataset_success(temp_datasets):
    text_dir, _ = temp_datasets
    rows = validate_text_dataset(text_dir)
    assert rows == 3

def test_validate_embedding_dataset_success(temp_datasets):
    _, emb_dir = temp_datasets
    rows = validate_embedding_dataset(emb_dir)
    assert rows == 3

def test_validate_text_dataset_invalid_columns(temp_datasets):
    text_dir, _ = temp_datasets
    invalid_table = pa.table({
        "wrong_id": ["1"],
        "text": ["hello"]
    })
    pq.write_table(invalid_table, text_dir / "invalid.parquet")
    
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_text_dataset(text_dir)

def test_validate_text_dataset_null_values(temp_datasets):
    text_dir, _ = temp_datasets
    # Clear existing valid files to avoid mixing
    for f in text_dir.glob("*.parquet"):
        f.unlink()
        
    null_table = pa.table({
        "id": ["1", None],
        "text": ["hello", "world"]
    })
    pq.write_table(null_table, text_dir / "nulls.parquet")
    
    with pytest.raises(ValueError, match="Found null values"):
        validate_text_dataset(text_dir)

def test_validate_embedding_dataset_mismatched_dims(temp_datasets):
    _, emb_dir = temp_datasets
    # Clear existing valid files
    for f in emb_dir.glob("*.parquet"):
        f.unlink()
        
    mismatched_table = pa.table({
        "document_id": ["1", "2"],
        "embedding": [[0.1, 0.2], [0.3, 0.4, 0.5]] # Different dimensions
    })
    pq.write_table(mismatched_table, emb_dir / "mismatched.parquet")
    
    with pytest.raises(ValueError, match="Inconsistent embedding size"):
        validate_embedding_dataset(emb_dir)
