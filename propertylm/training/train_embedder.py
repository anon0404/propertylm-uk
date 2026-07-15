from __future__ import annotations
from pathlib import Path
import polars as pl
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses


def train(pairs: Path, base_model: str, output: Path, epochs: int = 1, batch_size: int = 32):
    df = pl.read_parquet(pairs).to_pandas()
    dataset = Dataset.from_pandas(df[["anchor", "positive"]], preserve_index=False)
    model = SentenceTransformer(base_model)
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=2e-5,
        warmup_ratio=.1, fp16=False, bf16=False, save_strategy="epoch",
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
    model.save_pretrained(str(output))
