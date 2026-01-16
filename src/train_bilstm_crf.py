from __future__ import annotations

import argparse
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

from data_conll import load_conll2003_hf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", type=str, default="outputs/bilstm-crf.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=100)
    p.add_argument("--hid_dim", type=int, default=256)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(token_seqs: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    c = Counter(t for seq in token_seqs for t in seq)
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in c.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_seq(seq: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.get(t, vocab["<unk>"]) for t in seq][:max_len]
    mask = [1] * len(ids)
    # pad
    while len(ids) < max_len:
        ids.append(vocab["<pad>"])
        mask.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class ConllTorchDataset(Dataset):
    def __init__(self, tokens: List[List[str]], tags: List[List[int]], vocab: Dict[str, int], max_len: int):
        self.tokens = tokens
        self.tags = tags
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.tokens)

    def __getitem__(self, idx):
        x, mask = encode_seq(self.tokens[idx], self.vocab, self.max_len)
        y = self.tags[idx][:self.max_len]
        y = y + [0] * (self.max_len - len(y))  # pad label ids with 0 (O usually 0 in HF conll2003)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, mask


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int, hid_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, mask):
        z = self.emb(x)
        z, _ = self.lstm(z)
        emissions = self.fc(z)
        return emissions

    def loss(self, x, tags, mask):
        emissions = self.forward(x, mask)
        # CRF returns log-likelihood; we minimize negative log-likelihood
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_conll2003_hf()
    label_names = ds["train"].features["ner_tags"].feature.names
    num_tags = len(label_names)

    train_tokens = ds["train"]["tokens"]
    train_tags = ds["train"]["ner_tags"]
    val_tokens = ds["validation"]["tokens"]
    val_tags = ds["validation"]["ner_tags"]

    vocab = build_vocab(train_tokens, min_freq=1)

    train_ds = ConllTorchDataset(train_tokens, train_tags, vocab, args.max_len)
    val_ds = ConllTorchDataset(val_tokens, val_tags, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BiLSTMCRF(len(vocab), num_tags, args.emb_dim, args.hid_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            loss = model.loss(x, y, mask)
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                val_total += model.loss(x, y, mask).item()

        print(f"Epoch {epoch} | train_loss={total/len(train_loader):.4f} | val_loss={val_total/len(val_loader):.4f}")

        if val_total < best_val:
            best_val = val_total
            torch.save(
                {"state_dict": model.state_dict(), "vocab": vocab, "label_names": label_names, "args": vars(args)},
                args.output_path,
            )
            print("Saved best model ->", args.output_path)


if __name__ == "__main__":
    main()
