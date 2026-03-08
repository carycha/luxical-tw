import argparse
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

from luxical_tw.dataset_abstractions import ManyParquetFileDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def create_parser():
    parser = argparse.ArgumentParser(description="Validate training dataset format and consistency.")
    parser.add_argument("--text_path", type=str, required=True, help="Path to text parquet files directory.")
    parser.add_argument("--emb_path", type=str, required=True, help="Path to teacher embedding parquet files directory.")
    return parser

def validate_text_dataset(path: Path):
    """Validates the text dataset format."""
    logger.info(f"Validating text dataset at: {path}")
    files = list(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No parquet files found at {path}")

    dataset = ManyParquetFileDataset(files)
    expected_columns = {"id", "text"}
    
    total_rows = 0
    for batch in dataset.stream_record_batches():
        columns = set(batch.schema.names)
        if columns != expected_columns:
            raise ValueError(f"Invalid columns in text dataset: {columns}. Expected: {expected_columns}")
        
        # Check for nulls
        for col in expected_columns:
            if batch[col].null_count > 0:
                raise ValueError(f"Found null values in column '{col}' of text dataset.")
        
        total_rows += len(batch)
    
    logger.info(f"Text dataset validation passed. Total rows: {total_rows}")
    return total_rows

def validate_embedding_dataset(path: Path):
    """Validates the teacher embedding dataset format."""
    logger.info(f"Validating embedding dataset at: {path}")
    files = list(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No parquet files found at {path}")

    dataset = ManyParquetFileDataset(files)
    expected_columns = {"document_id", "embedding"}
    
    total_rows = 0
    embedding_size = None
    
    for batch in dataset.stream_record_batches():
        columns = set(batch.schema.names)
        if columns != expected_columns:
            raise ValueError(f"Invalid columns in embedding dataset: {columns}. Expected: {expected_columns}")
        
        # Check for nulls
        for col in expected_columns:
            if batch[col].null_count > 0:
                raise ValueError(f"Found null values in column '{col}' of embedding dataset.")
        
        # Check embedding size consistency
        embeddings = batch["embedding"]
        for i in range(len(embeddings)):
            current_size = len(embeddings[i])
            if embedding_size is None:
                embedding_size = current_size
            elif current_size != embedding_size:
                raise ValueError(f"Inconsistent embedding size: {current_size}. Expected: {embedding_size}")
        
        total_rows += len(batch)
    
    logger.info(f"Embedding dataset validation passed. Total rows: {total_rows}, Dimension: {embedding_size}")
    return total_rows

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    text_path = Path(args.text_path)
    emb_path = Path(args.emb_path)
    
    try:
        text_rows = validate_text_dataset(text_path)
        emb_rows = validate_embedding_dataset(emb_path)
        
        if text_rows != emb_rows:
            raise ValueError(f"Row count mismatch! Text rows: {text_rows}, Embedding rows: {emb_rows}")
        
        logger.info("Success: All validation checks passed.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
