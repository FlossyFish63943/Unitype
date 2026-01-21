# ==========================================
# Unitype — Tiny Unity C# Transformer
# ==========================================

import os
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# CLI
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--infer", action="store_true")
args = parser.parse_args()

if args.train == args.infer:
    raise RuntimeError("Use exactly one flag: --train OR --infer")

# ==========================================
# Files
# ==========================================
DATA_FILE = "data.txt"
VOCAB_FILE = "vocab.json"
MODEL_FILE = "unitype_fp32.pt"

# ==========================================
# Device
# ==========================================
DEVICE = "cpu"

# ==========================================
# LOCKED HYPERPARAMETERS
# ==========================================
BLOCK_SIZE = 256
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 8
EPOCHS = 600
LR = 3e-4

# ==========================================
# Vocabulary
# ==========================================
if args.train:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("data.txt missing")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)

    print(f"[Unitype] vocab size = {vocab_size}")

else:
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VOCAB_FILE):
        raise RuntimeError("Model or vocab missing. Train first.")

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    vocab_size = len(itos)

# ==========================================
# Transformer Building Blocks
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / math.sqrt(C)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = EMBED_DIM // NUM_HEADS
        self.heads = nn.ModuleList([Head(head_size) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ==========================================
# Model
# ==========================================
class Unitype(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new=400):
        for _ in range(max_new):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# ==========================================
# Init
# ==========================================
model = Unitype().to(DEVICE)

# ==========================================
# TRAIN
# ==========================================
if args.train:
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def get_batch():
        ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch} | loss {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print("[Unitype] Training complete")

# ==========================================
# INFER
# ==========================================
if args.infer:
    state = torch.load(MODEL_FILE, map_location=DEVICE)

    if state["token_emb.weight"].shape[0] != vocab_size:
        raise RuntimeError("Vocab mismatch — retrain required")

    model.load_state_dict(state)
    model.eval()

    print("\nUnitype ready. Type Unity C# prompts. 'exit' to quit.\n")

    while True:
        prompt = input("> ")
        if prompt.lower() == "exit":
            break

        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
        out = model.generate(idx)[0].tolist()
        print("".join(itos[i] for i in out))