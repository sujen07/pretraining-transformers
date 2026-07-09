from datasets import load_dataset
import os
from torch.utils.data import Dataset, DataLoader
import json
import torch
from tokenization.bpe import BPE, PAD_TOKEN

SEQ_LEN = 128

def initial_load():
    """
    Load the dataset from the Hugging Face dataset repository.
    """

    # Check if the dataset already exists
    name = "openwebtext-100k"
    if os.path.exists(f"{name}.jsonl"):
        print(f"Dataset {name} already exists")
        return

    ds = load_dataset("mychen76/openwebtext-100k", split='train')
    ds.to_json(f"{name}.jsonl", orient="records", lines=True)
    return ds

def create_dataset(dataset_path: str, bpe: BPE):
    """
    Create a dataset from the JSONL file.
    """
    with open(dataset_path, "r") as f:
        dataset = []
        for line in f:
            data = json.loads(line)
            text = data["text"]
            tokens = bpe.encode(text)
            dataset.append(tokens)
        return dataset


def _collate_fn(batch: list[list[int]], pad_id: int, seq_len: int = SEQ_LEN):
    """
    Stack sequences to a fixed length: truncate longer sequences and pad shorter ones.
    """
    batch_tensor = torch.full((len(batch), seq_len), pad_id, dtype=torch.long)
    for i, tokens in enumerate(batch):
        length = min(len(tokens), seq_len)
        batch_tensor[i, :length] = torch.tensor(tokens[:length], dtype=torch.long)
    return batch_tensor


def create_dataloader(dataset_path: str, bpe: BPE, batch_size: int, seq_len: int = SEQ_LEN):
    """
    Create a dataloader from the JSONL file.
    """
    dataset = create_dataset(dataset_path, bpe)
    pad_id = bpe.word_to_id[PAD_TOKEN]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_fn(batch, pad_id, seq_len),
    )
    return dataloader


if __name__ == "__main__":
    #initial_load()
    bpe = BPE(target_vocab_size=10000, chunk_size=10000, dataset_path="src/data/openwebtext-100k.jsonl", tokenizer_path="src/tokenization/bpe_tokenizer.json")
    dataset = create_dataset("src/data/openwebtext-100k.jsonl", bpe)
    dataloader = create_dataloader("src/data/openwebtext-100k.jsonl", bpe, batch_size=16, seq_len=SEQ_LEN)
    print(dataset[0])