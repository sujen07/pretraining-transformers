import torch
from models.transformer import Transformer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from preprocessing import create_dataloader
from tokenization.bpe import BPE, PAD_TOKEN

# Set True for a quick local end-to-end check. Set False on the GPU machine.
SMOKE_TEST = True

if SMOKE_TEST:
    BATCH_SIZE = 2
    SEQ_LEN = 64
    N_LAYERS = 2
    N_HEADS = 2
    D_MODEL = 64
    N_EPOCHS = 1
    MAX_LINES = 200          # only tokenize first N lines of the corpus
    MAX_TRAIN_STEPS = 5      # stop after this many optimizer steps
    MAX_VAL_STEPS = 2
else:
    BATCH_SIZE = 32
    SEQ_LEN = 128
    N_LAYERS = 12
    N_HEADS = 12
    D_MODEL = 768
    N_EPOCHS = 10
    MAX_LINES = None
    MAX_TRAIN_STEPS = None
    MAX_VAL_STEPS = None

N_CLASSES = 10000  # tokenizer target vocab; actual size comes from the saved file

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def plot_loss(train_loss_history: list[float], val_loss_history: list[float]):
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.legend()
    plt.xlabel("Step" if SMOKE_TEST else "Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.savefig("loss.png")


def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_vocab, device):
    device_type = device.type if device.type in ("cuda", "mps", "cpu") else "cpu"
    # bfloat16 on MPS/CUDA; float32 autocast is a no-op-ish path on CPU
    amp_dtype = torch.bfloat16 if device_type in ("cuda", "mps") else torch.float32

    train_loss_history = []
    val_loss_history = []
    for epoch in range(N_EPOCHS):
        for step, batch in enumerate(train_loader):
            if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
                break
            model.train()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                output = model(input_ids)
                output = output[:, :-1, :].reshape(-1, num_vocab)
                labels = labels.reshape(-1)
                loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_history.append(loss.item())
            print(f"  step {step + 1}: loss={loss.item():.4f}", flush=True)

        val_loss = evaluate(model, criterion, val_loader, num_vocab, device)
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        val_loss_history.append(val_loss)
    return train_loss_history, val_loss_history


def evaluate(model: Transformer, criterion, val_loader: DataLoader, num_vocab: int, device: torch.device):
    model.eval()
    device_type = device.type if device.type in ("cuda", "mps", "cpu") else "cpu"
    amp_dtype = torch.bfloat16 if device_type in ("cuda", "mps") else torch.float32
    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(val_loader):
            if MAX_VAL_STEPS is not None and step >= MAX_VAL_STEPS:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                output = model(input_ids)
                output = output[:, :-1, :].reshape(-1, num_vocab)
                labels = labels.reshape(-1)
                total_loss += criterion(output, labels).item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    print(f"Device: {DEVICE} | SMOKE_TEST={SMOKE_TEST}")
    tokenizer = BPE(
        target_vocab_size=N_CLASSES,
        chunk_size=10000,
        dataset_path="src/data/openwebtext-100k.jsonl",
        tokenizer_path="src/tokenization/bpe_tokenizer.json",
    )
    num_vocab = len(tokenizer.word_to_id)
    model = Transformer(D_MODEL, N_HEADS, N_LAYERS, num_vocab, SEQ_LEN).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[PAD_TOKEN])
    optimizer = Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader = create_dataloader(
        "src/data/openwebtext-100k.txt",
        tokenizer,
        BATCH_SIZE,
        SEQ_LEN,
        max_lines=MAX_LINES,
    )
    steps_per_epoch = len(train_loader) if MAX_TRAIN_STEPS is None else min(MAX_TRAIN_STEPS, len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(N_EPOCHS * steps_per_epoch, 1))
    train_loss_history, val_loss_history = train(
        model, criterion, train_loader, val_loader, optimizer, scheduler, num_vocab, DEVICE
    )
    plot_loss(train_loss_history, val_loss_history)
    print("Done. Wrote loss.png")


if __name__ == "__main__":
    main()
