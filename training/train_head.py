"""Phase 2, step 3 — trainable head over frozen patch tokens (TAP-style).

A single learned attention query pools the patch tokens, concatenated with the
CLS token, into a binary logit. A few hundred K parameters.

Memory design (learned the hard way — 30K samples of fp32 patch tokens is ~12 GB
and OOM-kills a Colab session): shards stay fp16 in ONE preallocated host buffer
(no concatenate doubling); only each minibatch is converted to fp32, on the GPU.
Peak host RAM ≈ dataset-in-fp16 + one shard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _lazy_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_head(dim: int):
    torch, nn = _lazy_torch()

    class AttnPoolHead(nn.Module):
        def __init__(self, dim: int, hidden: int = 256):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
            self.mlp = nn.Sequential(
                nn.LayerNorm(2 * dim),
                nn.Linear(2 * dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, patch, cls):
            q = self.query.expand(patch.shape[0], -1, -1)
            pooled, _ = self.attn(q, patch, patch)
            z = torch.cat([pooled.squeeze(1), cls], dim=-1)
            return self.mlp(z).squeeze(-1)  # logits

    return AttnPoolHead(dim)


def _load_split(emb_dir: str | Path):
    """Load all shards into preallocated fp16 arrays. Peak RAM ≈ total + 1 shard."""
    shards = sorted(Path(emb_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz files in {emb_dir}")

    # Pass 1: row counts (decompresses only the tiny 'label' member per shard)
    counts = []
    for s in shards:
        with np.load(s) as d:
            counts.append(len(d["label"]))
    with np.load(shards[0]) as d:
        n_tok, dim = d["patch"].shape[1], d["patch"].shape[2]

    total = sum(counts)
    patch = np.empty((total, n_tok, dim), dtype=np.float16)
    cls = np.empty((total, dim), dtype=np.float16)
    y = np.empty(total, dtype=np.float32)

    # Pass 2: fill, one shard resident at a time
    pos = 0
    for s, n in zip(shards, counts):
        with np.load(s) as d:
            patch[pos : pos + n] = d["patch"]
            cls[pos : pos + n] = d["cls"]
            y[pos : pos + n] = d["label"]
        pos += n
    return patch, cls, y


def _batched_logits(model, patch, cls, device, bs: int = 512):
    torch, _ = _lazy_torch()
    outs = []
    with torch.no_grad():
        for i in range(0, len(patch), bs):
            p = torch.from_numpy(patch[i : i + bs]).to(device).float()
            c = torch.from_numpy(cls[i : i + bs]).to(device).float()
            outs.append(model(p, c).cpu())
    return torch.cat(outs)


def train(
    train_dir: str | Path,
    val_dir: str | Path,
    out: str | Path = "head.pt",
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
):
    torch, nn = _lazy_torch()

    ptr, ctr, ytr = _load_split(train_dir)   # fp16 host buffers
    pva, cva, yva = _load_split(val_dir)
    dim = ptr.shape[-1]
    print(f"train: {len(ytr)} samples ({ptr.nbytes/1e9:.1f} GB fp16), "
          f"val: {len(yva)} samples, dim={dim}, device={device}")

    model = build_head(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(ytr)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            # fp16 host -> fp32 GPU, one batch at a time (~100 MB, not 12 GB)
            p = torch.from_numpy(ptr[idx]).to(device).float()
            c = torch.from_numpy(ctr[idx]).to(device).float()
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(p, c), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        logits = _batched_logits(model, pva, cva, device)
        acc = float(((logits > 0).numpy() == (yva == 1)).mean())
        print(f"epoch {epoch}: train_loss={total/n:.4f} val_acc={acc:.4f}")

    torch.save({"state_dict": model.state_dict(), "dim": dim}, out)
    np.savez(Path(out).with_suffix(".val_logits.npz"),
             logits=logits.numpy(), label=yva)
    return out
