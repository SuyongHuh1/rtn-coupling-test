"""R7 network: dilated residual TCN over [first difference, detrended level] with global pooling.

Heads: coupling logit (main), number of traps (classes K = 2, 3, 4), shared-drive logit.
Long records are scored as the mean coupling logit over non-overlapping windows of the
training length, so a record's score does not depend on its length.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, C, d):
        super().__init__()
        self.c1 = nn.Conv1d(C, C, 3, dilation=d, padding=d)
        self.c2 = nn.Conv1d(C, C, 1)
        self.n = nn.GroupNorm(8, C)
        self.a = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.a(self.n(self.c1(x))))


class CouplingNet(nn.Module):
    def __init__(self, C=64, n_dil=10, c_in=2):
        super().__init__()
        self.inp = nn.Conv1d(c_in, C, 5, padding=2)
        self.blocks = nn.Sequential(*[Block(C, 2 ** i) for i in range(n_dil)])
        self.head = nn.Sequential(nn.Linear(3 * C, 128), nn.GELU(), nn.Linear(128, 1 + 3 + 1 + 4 * 3 + 6))

    def features(self, x):
        h = self.blocks(self.inp(x))
        return torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], -1)

    def forward(self, x):
        o = self.head(self.features(x))
        return dict(coupled=o[:, 0], K=o[:, 1:4], drive=o[:, 4], theta=o[:, 5:17].view(-1, 4, 3), Jvec=o[:, 17:23])


@torch.no_grad()
def score_record(model, y, win=4096, device="cpu"):
    """y: 1-D float tensor (already rescaled in time). Mean coupling logit over windows."""
    from .sim2_torch import preprocess
    n = max(1, y.numel() // win)
    segs = y[: n * win].view(n, win) if y.numel() >= win else y[None]
    x = preprocess(segs.float().to(device))
    out = model(x)
    return dict(coupled=float(out["coupled"].mean()), drive=float(out["drive"].mean()),
                K=out["K"].softmax(-1).mean(0).cpu().tolist(),
                theta=out["theta"].mean(0).cpu().tolist(), Jvec=out["Jvec"].mean(0).cpu().tolist())
