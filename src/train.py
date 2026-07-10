import torch
from models.transformer import Transformer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from preprocessing import create_dataloader
from tokenization.bpe import BPE, UNK_TOKEN, PAD_TOKEN

# Hyperparameters
N_CLASSES = 10000
BATCH_SIZE = 128
SEQ_LEN = 128
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
N_EPOCHS = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_loss(loss_history: list[float]):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.savefig("loss.png")
    plt.show()

def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_vocab, device):
    model.train()
    loss_history = []
    for epoch in range(N_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids)
            output = output.view(-1, num_vocab)
            labels = labels.view(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
        val_loss = evaluate(model, criterion, val_loader, num_vocab, device)
        print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
        loss_history.append(val_loss.item())
    return loss_history

def evaluate(model: Transformer, criterion, val_loader: DataLoader, num_vocab: int, device: torch.device):
    model.eval()
    torch.no_grad()
    loss = 0
    num_batches = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids)
        output = output.view(-1, num_vocab)
        labels = labels.view(-1)
        loss += criterion(output, labels)
        num_batches += 1

    return loss / num_batches

def main():
    tokenizer = BPE(target_vocab_size=N_CLASSES, chunk_size=10000, dataset_path="src/data/openwebtext-100k.jsonl", tokenizer_path="src/tokenization/bpe_tokenizer.json")
    num_vocab = len(tokenizer.word_to_id)
    model = Transformer(D_MODEL, N_HEADS, N_LAYERS, num_vocab, SEQ_LEN).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[PAD_TOKEN])
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    train_loader, val_loader = create_dataloader("src/data/openwebtext-100k.jsonl", tokenizer, BATCH_SIZE, SEQ_LEN)
    loss_history = train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_vocab, DEVICE)
    plot_loss(loss_history)

if __name__ == "__main__":
    main()