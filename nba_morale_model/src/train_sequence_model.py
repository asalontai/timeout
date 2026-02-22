import math
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR


class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sequences(df, seq_len=50, step=5):
    features = [
        "momentum_index",
        "momentum_shift",
        "gsw_back_to_back_3s_3min",
        "opp_back_to_back_3s_3min",
        "gsw_score",
        "opp_score",
        "is_gsw",
    ]

    sequences = []
    labels = []

    for game_id, g in df.groupby("game_id"):
        g = g.sort_values("event_num")
        X = g[features].fillna(0).to_numpy(dtype=float)
        y = int(g["gsw_win"].iloc[0])

        for end in range(seq_len, len(X), step):
            seq = X[end - seq_len:end]
            sequences.append(seq)
            labels.append(y)

    return sequences, labels


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits.squeeze(-1)


def train_and_eval(train_df, test_df, seq_len=50):
    train_seqs, train_labels = build_sequences(train_df, seq_len=seq_len)
    test_seqs, test_labels = build_sequences(test_df, seq_len=seq_len)

    if not train_seqs or not test_seqs:
        raise RuntimeError("Not enough sequences. Try reducing seq_len or adding more seasons.")

    train_ds = SeqDataset(torch.tensor(train_seqs, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    test_ds = SeqDataset(torch.tensor(test_seqs, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = LSTMModel(input_dim=train_ds[0][0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

    print(f"Playoff test accuracy: {correct/total:.3f}")


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play.csv")

    # Determine game outcome from final score in PBP
    final_scores = df.sort_values("event_num").groupby("game_id").tail(1)
    df = df.merge(final_scores[["game_id", "gsw_score", "opp_score"]], on="game_id", suffixes=("", "_final"))
    df["gsw_win"] = (df["gsw_score_final"] > df["opp_score_final"]).astype(int)

    # Train on regular season, test on playoffs
    train_df = df[df["season_type"] == "regular"].copy()
    test_df = df[df["season_type"] == "playoffs"].copy()

    train_and_eval(train_df, test_df, seq_len=50)


if __name__ == "__main__":
    main()
