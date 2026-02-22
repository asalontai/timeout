import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import DATA_DIR


def classify_event(desc: str, is_team: int, fine: bool = False) -> str:
    d = (desc or "").lower()
    team = "TEAM" if is_team == 1 else "OPP"

    if "turnover" in d:
        return f"{team}_TO"
    if "3-pt" in d or "3pt" in d:
        if fine and "miss" in d:
            return f"{team}_3MISS"
        return f"{team}_3"
    if "free throw" in d:
        if fine and "miss" in d:
            return f"{team}_FTMISS"
        return f"{team}_FT"
    if any(k in d for k in ["layup", "dunk", "jumper", "shot", "tip-in", "hook"]):
        if fine and "miss" in d:
            return f"{team}_2MISS"
        return f"{team}_2"
    if "foul" in d:
        return f"{team}_FOUL"
    if "rebound" in d:
        return f"{team}_REB"
    if "steal" in d:
        return f"{team}_STL"
    if "block" in d:
        return f"{team}_BLK"

    return f"{team}_OTHER"


class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sequences(df, vocab, seq_len=20, fine=False):
    sequences = []
    labels = []

    for game_id, g in df.groupby("game_id"):
        g = g.sort_values("event_num")
        tokens = [classify_event(d, i, fine=fine) for d, i in zip(g["description"], g["is_team"])]

        token_ids = [vocab[t] for t in tokens]

        for i in range(seq_len, len(token_ids)):
            seq = token_ids[i - seq_len:i]
            label = token_ids[i]
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


class NextPlayModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def train_model(train_ds, test_ds, vocab_size, epochs=5):
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = NextPlayModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    # eval top-1 accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    print(f"Top-1 accuracy: {correct/total:.3f}")
    return model


def run(fine=False):
    df = pd.read_csv(DATA_DIR / "momentum_per_play_allteams.csv")

    tokens = [classify_event(d, i, fine=fine) for d, i in zip(df["description"], df["is_team"])]
    vocab = {t: idx for idx, t in enumerate(sorted(set(tokens)))}

    sequences, labels = build_sequences(df, vocab, seq_len=20, fine=fine)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    train_ds = SeqDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    test_ds = SeqDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

    print(f"Vocab size: {len(vocab)} | Samples: {len(sequences)} | Fine={fine}")
    model = train_model(train_ds, test_ds, vocab_size=len(vocab))

    # Save
    kind = "fine" if fine else "coarse"
    model_path = DATA_DIR / f"next_play_model_{kind}.pt"
    vocab_path = DATA_DIR / f"next_play_vocab_{kind}.json"
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Saved model to {model_path}")
    print(f"Saved vocab to {vocab_path}")


if __name__ == "__main__":
    print("Training coarse model...")
    run(fine=False)
    print("\nTraining fine-grained model...")
    run(fine=True)
