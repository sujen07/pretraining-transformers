from __future__ import annotations

from datasets import load_dataset
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch
from tokenization.bpe import BPE, PAD_TOKEN, EOS_TOKEN

SEQ_LEN = 128
# Flush the token buffer to disk this often while building the cache, so we never
# hold the whole tokenized corpus in a Python list.
_CACHE_FLUSH_EVERY = 1_000_000


def initial_load():
    """
    Load the dataset from the Hugging Face dataset repository.
    """

    # Check if the dataset already exists
    name = "openwebtext-100k"
    if os.path.exists(f"{name}.txt"):
        print(f"Dataset {name} already exists")
        return

    ds = load_dataset("mychen76/openwebtext-100k", split='train')

    # Convert the dataset to one text file
    with open(f"data/{name}.txt", "w") as f:
        for item in ds:
            f.write(item["text"] + EOS_TOKEN.decode())

    return ds

def _cache_path(dataset_path: str, bpe: BPE, max_lines: int | None = None) -> str:
    """
    Path of the token cache for this (dataset, tokenizer) pair. The vocab size is
    baked into the name so a re-trained tokenizer never silently reuses a stale cache.
    """
    base, _ = os.path.splitext(dataset_path)
    suffix = f".lines{max_lines}" if max_lines is not None else ""
    return f"{base}.tok{len(bpe.word_to_id)}{suffix}.bin"


def _build_token_cache(
    dataset_path: str, bpe: BPE, cache_path: str, max_lines: int | None = None
) -> None:
    """
    Encode the corpus once and stream it to disk as a flat uint16 array.
    If max_lines is set, only the first N lines are tokenized (for smoke tests).
    """
    assert len(bpe.word_to_id) <= np.iinfo(np.uint16).max + 1, \
        "Vocab too large for uint16 cache; widen the dtype to uint32."
    tmp_path = cache_path + ".tmp"
    buf = []
    n_lines = 0
    with open(dataset_path, "r") as f, open(tmp_path, "wb") as out:
        for line in f:
            buf.extend(bpe.encode(line))
            n_lines += 1
            if len(buf) >= _CACHE_FLUSH_EVERY:
                np.asarray(buf, dtype=np.uint16).tofile(out)
                buf.clear()
                print(f"  cached {n_lines} lines...", flush=True)
            if max_lines is not None and n_lines >= max_lines:
                break
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(out)
    os.replace(tmp_path, cache_path)  # atomic: a crash mid-build leaves no half cache
    print(f"Token cache ready ({n_lines} lines) -> {cache_path}")


def load_tokens(
    dataset_path: str, bpe: BPE, max_lines: int | None = None
) -> np.memmap:
    """
    Return the tokenized corpus as a read-only memmap, building the cache on first
    use and rebuilding it whenever the source file is newer than the cache.
    """
    cache_path = _cache_path(dataset_path, bpe, max_lines)
    # Rebuild if either the source text or the tokenizer changed after the cache
    # was written (retraining to the same vocab size keeps the same cache name).
    sources = [dataset_path, bpe.tokenizer_path]
    fresh = os.path.exists(cache_path) and all(
        os.path.getmtime(cache_path) >= os.path.getmtime(s)
        for s in sources if os.path.exists(s)
    )
    if not fresh:
        print(f"Building token cache -> {cache_path}")
        _build_token_cache(dataset_path, bpe, cache_path, max_lines=max_lines)
    else:
        print(f"Using cached tokens <- {cache_path}")
    return np.memmap(cache_path, dtype=np.uint16, mode="r")


class TokenDataset(Dataset):
    """
    Non-overlapping SEQ_LEN windows over the flat token stream, read straight from
    the memmap. No per-sequence Python list is materialized.
    """

    def __init__(self, tokens: np.memmap, seq_len: int = SEQ_LEN):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = len(tokens) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> np.ndarray:
        start = idx * self.seq_len
        return np.asarray(self.tokens[start : start + self.seq_len])


def create_dataset(
    dataset_path: str,
    bpe: BPE,
    seq_len: int = SEQ_LEN,
    max_lines: int | None = None,
):
    """
    Create a memmap-backed dataset of seq_len token windows.
    """
    return TokenDataset(load_tokens(dataset_path, bpe, max_lines=max_lines), seq_len)


def _collate_fn(batch: list[np.ndarray]):
    """
    Stack a list of token windows into a batch and build shifted next-token labels.
    """
    batch_tensor = torch.from_numpy(np.stack(batch).astype(np.int64))
    labels = batch_tensor[:, 1:].clone()
    return {"input_ids": batch_tensor, "labels": labels}


def create_dataloader(
    dataset_path: str,
    bpe: BPE,
    batch_size: int,
    seq_len: int = SEQ_LEN,
    max_lines: int | None = None,
):
    """
    Create train/val dataloaders from a text corpus.
    """
    dataset = create_dataset(dataset_path, bpe, seq_len=seq_len, max_lines=max_lines)
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty after tokenization (seq_len={seq_len}, max_lines={max_lines}). "
            "Increase max_lines or lower seq_len."
        )
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    #initial_load()
    bpe = BPE(target_vocab_size=10000, chunk_size=10000, dataset_path="src/data/openwebtext-100k.jsonl", tokenizer_path="src/tokenization/bpe_tokenizer.json")
    dataset = create_dataset("src/data/openwebtext-100k.jsonl", bpe)
    train_dataloader, val_dataloader = create_dataloader("src/data/openwebtext-100k.jsonl", bpe, batch_size=16, seq_len=SEQ_LEN)
    print(dataset[0])