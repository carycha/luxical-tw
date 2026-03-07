from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence, Literal
from time import perf_counter
from contextlib import contextmanager

import torch
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luxical.embedder import Embedder, Model, initialize_embedder_from_ngram_summary
from luxical.training import (
    dataloader, 
    contrastive_distillation_loss, 
    wsd_lr_schedule, 
    equal_beta_adamw
)
from luxical.dataset_abstractions import ManyParquetFileDataset
from luxical.tokenization import load_arrow_tokenizer_from_pretrained
from luxical.ngrams import SpaceSavingNgramSummary
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder

logger = logging.getLogger(__name__)

@contextmanager
def _time_section(timing_dict: dict[str, float], name: str):
    start = perf_counter()
    try:
        yield None
    finally:
        timing_dict[name] = perf_counter() - start

class Trainer:
    """
    High-level Trainer API for Luxical-TW.
    Encapsulates the training loop, dataloading, and optimization.
    """
    def __init__(
        self,
        model: Model,
        lr: float = 1e-2,
        batch_size: int = 4096,
        loss_temperature: float = 3.0,
        device: str = "cpu",
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.loss_temperature = loss_temperature
        self.device = torch.device(device)
        
        # Prepare the student model for torch training
        self.student_nn = model.bow_to_dense_embedder.to_torch().to(self.device)
        self.optimizer = equal_beta_adamw(
            self.student_nn.parameters(), 
            lr=self.lr, 
            weight_decay=weight_decay
        )

    @classmethod
    def from_ngram_summary(
        cls,
        ngram_summary_path: str | Path,
        tokenizer_id: str = "google-bert/bert-base-uncased",
        sparse_to_dense_embedder_dims: tuple[int, ...] = (96, 3072, 3072, 192),
        min_ngram_count_multiple: float = 8.0,
        **trainer_kwargs
    ) -> Trainer:
        """Initialize a Trainer and a new Model from an ngram summary file."""
        logger.info(f"Initializing model from ngram summary: {ngram_summary_path}")
        ngram_summary = SpaceSavingNgramSummary.load_npz(ngram_summary_path)
        tokenizer = load_arrow_tokenizer_from_pretrained(tokenizer_id)
        
        model = initialize_embedder_from_ngram_summary(
            ngram_summary=ngram_summary,
            tokenizer=tokenizer,
            sparse_to_dense_embedder_dims=sparse_to_dense_embedder_dims,
            min_ngram_count_multiple=min_ngram_count_multiple,
        )
        return cls(model=model, **trainer_kwargs)

    def train(
        self,
        text_dataset_path: str | Path,
        teacher_emb_dataset_path: str | Path,
        num_epochs: float = 1.0,
        warmup_fraction: float = 0.05,
        decay_fraction: float = 0.1,
        teacher_emb_quantization_limit: float = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> list[float]:
        """
        Execute the training loop.
        """
        text_dataset = ManyParquetFileDataset.from_path(str(text_dataset_path))
        emb_dataset = ManyParquetFileDataset.from_path(str(teacher_emb_dataset_path))
        
        num_examples_per_epoch = self.batch_size * (len(text_dataset) // self.batch_size)
        num_total_examples = int(num_epochs * num_examples_per_epoch)
        num_steps = num_total_examples // self.batch_size
        num_warmup = int(num_steps * warmup_fraction)
        num_decay = int(num_steps * decay_fraction)

        logger.info(f"Starting training: {num_steps} steps, {num_epochs} epochs")
        
        loader = dataloader(
            text_dataset=text_dataset,
            teacher_emb_dataset=emb_dataset,
            teacher_emb_quantization_limit=teacher_emb_quantization_limit,
            batch_size=self.batch_size,
            num_batches=num_steps,
            streaming_shuffle_buffer_size=8 * self.batch_size,
        )
        
        losses = []
        self.student_nn.train()
        
        with logging_redirect_tqdm(), tqdm(total=num_total_examples, unit="ex", unit_scale=True) as pbar:
            for step_idx, (texts, teacher_embs) in enumerate(loader):
                step = step_idx + 1
                
                # Update learning rate
                current_lr = self.lr * wsd_lr_schedule(step, num_steps, num_warmup, num_decay)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                timing = {}
                with _time_section(timing, "preprocess"):
                    # Student side pipeline
                    tokens = self.model.tokenize(texts)
                    bow = self.model.bow_from_tokens(tokens)
                    tfidf = self.model.tfidf_from_bow(bow)
                    # Convert to torch (assuming helper exists or using SparseToDenseEmbedder bridge)
                    # For performance, we want this to be efficient
                
                self.optimizer.zero_grad()
                
                with _time_section(timing, "forward"):
                    student_embs = self.student_nn(tfidf)
                
                with _time_section(timing, "loss"):
                    teacher_embs = teacher_embs.to(self.device)
                    loss = contrastive_distillation_loss(
                        student_embs, teacher_embs, self.loss_temperature
                    )
                
                with _time_section(timing, "backward"):
                    loss.backward()
                    self.optimizer.step()
                
                loss_val = loss.item()
                losses.append(loss_val)
                pbar.update(len(texts))
                
                if step % 10 == 0 or step == 1:
                    timing_msg = " | ".join(f"{k}: {v:.2f}s" for k, v in timing.items())
                    logger.info(f"Step {step}/{num_steps} | Loss: {loss_val:.6f} | LR: {current_lr:.6f} | {timing_msg}")
                    
                if checkpoint_path and step % 500 == 0:
                    self.save(checkpoint_path)

        return losses

    def save(self, path: str | Path) -> None:
        """
        Sync weights and save the model.
        """
        # Create a new embedder component with updated weights
        new_nn = SparseToDenseEmbedder.from_torch(self.student_nn)
        self.model = self.model.replace_sparse_to_dense_embedder(new_nn)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
