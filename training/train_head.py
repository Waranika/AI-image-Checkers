"""Phase 2, step 3 — trainable head over frozen patch tokens (TAP-style).

A single learned attention query pools the patch tokens, concatenated with the
CLS token, into a binary logit. A few hundred K parameters: trains in minutes on
CPU once embeddings are precomputed.

Colab usage:
    from training.train_head import train
    ckpt = train("emb/train", "emb/val", out="head.pt", epochs=5)
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


def _load_shards(emb_dir: str | Path):
    shards = sorted(Path(emb_dir).glob("shard_*.npz"))
    for s in shards:
        d = np.load(s)
        yield d["patch"].astype(np.float32), d["cls"].astype(np.float32), d["label"]


def train(
    train_dir: str | Path,
    val_dir: str | Path,
    out: str | Path = "head.pt",
    epochs: int = 5,
    lr: float = 3e-4,
    device: str = "cpu",
):
    torch, nn = _lazy_torch()

    # Small/medium datasets fit in RAM as fp32 arrays; stream shards otherwise.
    def load_all(d):
        ps, cs, ys = zip(*_load_shards(d))
        return (np.concatenate(ps), np.concatenate(cs), np.concatenate(ys))

    ptr, ctr, ytr = load_all(train_dir)
    pva, cva, yva = load_all(val_dir)
    dim = ptr.shape[-1]

    model = build_head(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(ytr)
    bs = 256
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            patch = torch.from_numpy(ptr[idx]).to(device)
            cls = torch.from_numpy(ctr[idx]).to(device)
            y = torch.from_numpy(ytr[idx].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(patch, cls), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)

        # validation
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(pva).to(device), torch.from_numpy(cva).to(device))
            acc = float(((logits > 0).cpu().numpy() == (yva == 1)).mean())
        print(f"epoch {epoch}: train_loss={total/n:.4f} val_acc={acc:.4f}")

    torch.save({"state_dict": model.state_dict(), "dim": dim}, out)

    # export validation logits for calibration
    np.savez(Path(out).with_suffix(".val_logits.npz"),
             logits=logits.cpu().numpy(), label=yva)
    return out
