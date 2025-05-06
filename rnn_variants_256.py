import os
import math
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

# -----------------------------------------
# 1. Hyperparameters
# -----------------------------------------
HIDDEN_SIZE = 256  

BATCH_SIZE   = 128
SEQ_LEN      = 50
EMBED_SIZE   = 128
CP_RANKS     = [50, 101, 350, 1500]
EPOCHS       = 30
LR           = 5e-4
CLIP         = 0.25
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# 2. Dataset loading and processing
# -----------------------------------------
os.makedirs("data", exist_ok=True)
for split in ("train", "valid", "test"):
    path = f"data/ptb.{split}.txt"
    if not os.path.exists(path):
        url = f"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{split}.txt"
        urllib.request.urlretrieve(url, path)

def read_text(split):
    with open(f"data/ptb.{split}.txt", "r", encoding="utf-8") as f:
        return f.read() + "\n"

train_text = read_text("train")
valid_text = read_text("valid")
test_text  = read_text("test")

chars = sorted(set(train_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)

def encode(text):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

train_data = encode(train_text)
valid_data = encode(valid_text)
test_data  = encode(test_text)

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i:i+self.seq_len]
        y = self.data[i+1:i+self.seq_len+1]
        return x, y

train_loader = DataLoader(CharDataset(train_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_loader = DataLoader(CharDataset(valid_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader  = DataLoader(CharDataset(test_data,  SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# -----------------------------------------
# 3. Model Definitions
# -----------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.dec = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h=None):
        e = self.emb(x)
        out, h = self.rnn(e, h)
        return self.dec(out), h

class CharMIRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.U = nn.Linear(embed_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta1 = nn.Parameter(torch.ones(hidden_size))
        self.beta2 = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.dec = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h=None):
        B, L = x.size()
        e = self.emb(x)
        h_t = torch.zeros(B, self.V.out_features, device=e.device) if h is None else h.squeeze(0)
        outputs = []
        for t in range(L):
            x_t = e[:, t, :]
            Vh = self.V(h_t)
            Ux = self.U(x_t)
            m = self.alpha * (Vh * Ux) + self.beta1 * Vh + self.beta2 * Ux + self.bias
            h_t = torch.tanh(m)
            outputs.append(h_t)
        out = torch.stack(outputs, dim=1)
        return self.dec(out), h_t.unsqueeze(0)

class Char2RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.A = nn.Parameter(torch.randn(hidden_size, hidden_size, embed_size) * 0.1)
        self.U = nn.Linear(embed_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.dec = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h=None):
        B, L = x.size()
        e = self.emb(x)
        h_t = torch.zeros(B, self.A.size(0), device=e.device) if h is None else h.squeeze(0)
        outputs = []
        for t in range(L):
            x_t = e[:, t, :]
            bil = torch.einsum('bh,hhe,be->bh', h_t, self.A, x_t)
            lin = self.V(h_t) + self.U(x_t) + self.bias
            h_t = torch.tanh(bil + lin)
            outputs.append(h_t)
        out = torch.stack(outputs, dim=1)
        return self.dec(out), h_t.unsqueeze(0)

class CharCPRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rank):
        super().__init__()
        H, E, R = hidden_size, embed_size, rank
        self.emb = nn.Embedding(vocab_size, E)
        self.Fh = nn.Parameter(torch.randn(H, R) * 0.1)
        self.Fx = nn.Parameter(torch.randn(E, R) * 0.1)
        self.Fo = nn.Parameter(torch.randn(H, R) * 0.1)
        self.U = nn.Linear(E, H, bias=False)
        self.V = nn.Linear(H, H, bias=False)
        self.bias = nn.Parameter(torch.zeros(H))
        self.dec = nn.Linear(H, vocab_size)
    def forward(self, x, h=None):
        B, L = x.size()
        e = self.emb(x)
        h_t = torch.zeros(B, self.Fh.size(0), device=e.device) if h is None else h.squeeze(0)
        outputs = []
        for t in range(L):
            x_t = e[:, t, :]
            u = h_t @ self.Fh
            v = x_t @ self.Fx
            bil = (u * v) @ self.Fo.t()
            lin = self.V(h_t) + self.U(x_t) + self.bias
            h_t = torch.tanh(bil + lin)
            outputs.append(h_t)
        out = torch.stack(outputs, dim=1)
        return self.dec(out), h_t.unsqueeze(0)

# -----------------------------------------
# 4. Training functions
# -----------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    h = None
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, h = model(x, h)
            h = h.detach()
            B, L, V = logits.size()
            loss = criterion(logits.view(B*L, V), y.view(B*L))
            total_loss += loss.item() * B * L
            total_tokens += B * L
    return (total_loss / total_tokens) / math.log(2)  # BPC

def train_model(model, name):
    print(f"\n========== Training {name} ==========")
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS+1):
        model.train()
        h = None
        total_loss, total_tokens = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, h = model(x, h)
            h = h.detach()
            B, L, V = logits.size()
            loss = criterion(logits.view(B*L, V), y.view(B*L))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item() * B * L
            total_tokens += B * L
        train_bpc = (total_loss / total_tokens) / math.log(2)
        val_bpc = evaluate(model, valid_loader)
        print(f"{name} | Epoch {epoch:2d} | Train BPC {train_bpc:.3f} | Valid BPC {val_bpc:.3f}")
    test_bpc = evaluate(model, test_loader)
    print(f"→ {name} final Test BPC: {test_bpc:.3f}")
    return test_bpc

# -----------------------------------------
# 5. Main code: Train & Save
# -----------------------------------------
results = {}

print(f"\n🔥 Training for HIDDEN_SIZE = {HIDDEN_SIZE}")

results[f"RNN_{HIDDEN_SIZE}"] = train_model(CharRNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE), f"RNN_{HIDDEN_SIZE}")
results[f"MIRNN_{HIDDEN_SIZE}"] = train_model(CharMIRNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE), f"MIRNN_{HIDDEN_SIZE}")
results[f"2RNN_{HIDDEN_SIZE}"] = train_model(Char2RNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE), f"2RNN_{HIDDEN_SIZE}")

for R in CP_RANKS:
    results[f"CPRNN_{HIDDEN_SIZE}_R{R}"] = train_model(CharCPRNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, R), f"CPRNN_{HIDDEN_SIZE}_R{R}")

# Save to JSON
save_path = f'results_hidden{HIDDEN_SIZE}.json'
with open(save_path, 'w') as f:
    json.dump(results, f)

print(f"\n✅ Results saved successfully to {save_path}")