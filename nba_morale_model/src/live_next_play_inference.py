import sys
import json
import torch
from torch import nn

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


def main():
    fine = "--fine" in sys.argv
    kind = "fine" if fine else "coarse"

    vocab_path = DATA_DIR / f"next_play_vocab_{kind}.json"
    model_path = DATA_DIR / f"next_play_model_{kind}.pt"

    if not vocab_path.exists() or not model_path.exists():
        raise RuntimeError("Model/vocab not found. Run next_play_sequence_model.py first.")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    model = NextPlayModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    seq_len = 20
    buffer = []

    print("Enter plays as: is_team<TAB>description (e.g., '1\tCurry makes 3-pt jumper')")
    print("Type 'quit' to exit.")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        try:
            is_team_str, desc = line.split("\t", 1)
            is_team = int(is_team_str)
        except Exception:
            print("Invalid format. Use: is_team<TAB>description")
            continue

        token = classify_event(desc, is_team, fine=fine)
        if token not in vocab:
            print(f"Unknown token: {token}")
            continue

        buffer.append(vocab[token])
        if len(buffer) < seq_len:
            print(f"Need {seq_len - len(buffer)} more plays...")
            continue
        if len(buffer) > seq_len:
            buffer = buffer[-seq_len:]

        x = torch.tensor(buffer, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=0)
            topk = torch.topk(probs, k=3)

        print("Top-3 next play predictions:")
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            print(f"  {inv_vocab[idx]}: {p:.3f}")


if __name__ == "__main__":
    main()
