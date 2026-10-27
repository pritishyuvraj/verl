"""Pre-rendered canonical Gemma tokens for verl's native SFT engine.

The stock multi-turn dataset renders each turn independently. Gemma renders
tool observations by looking ahead inside an assistant block, so this adapter
accepts whole-conversation tokens/masks audited by the caller's preprocessing.
This loader never renders, truncates, or certifies the caller's source data.
"""

from __future__ import annotations

import math


class GemmaMaskedSFTDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import pyarrow.parquet as pq

        if config.get("pad_mode") != "no_padding" or config.get("truncation") != "error":
            raise ValueError("Use nested no_padding batches with explicit length rejection")
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.real_count = len(self.rows)
        if not self.rows:
            raise ValueError("Empty prepared SFT dataset")
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define a neutral padding token")
        self.batch_size = int(config.train_batch_size)
        if self.batch_size < 1:
            raise ValueError("Global batch size must be positive")
        self.padded_count = math.ceil(self.real_count / self.batch_size) * self.batch_size
        for row in self.rows:
            ids, mask = row["input_ids"], row["loss_mask"]
            if (
                not 1 < len(ids) <= int(config.max_length)
                or len(ids) != len(mask)
                or mask[0] != 0
                or not any(mask)
                or any(type(token) is not int or token < 0 for token in ids)
                or any(type(value) is not int or value not in (0, 1) for value in mask)
            ):
                raise ValueError("Invalid canonical tokens or next-token loss mask")

    def __len__(self):
        return self.padded_count

    def __getitem__(self, index):
        import torch

        # Fill only incomplete global batches with explicit zero-loss neutral
        # rows. This retains every validation record exactly once across ranks.
        row = (
            self.rows[index]
            if index < self.real_count
            else {"input_ids": [self.pad_id, self.pad_id], "loss_mask": [0, 0]}
        )
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "position_ids": torch.arange(len(row["input_ids"]), dtype=torch.long),
            "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.long),
        }
