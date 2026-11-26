"""R9a: per-sample step detector / trap-signal denoiser.

Input  : record (any length), normalised per record by its white-noise scale (median |diff|).
Output : per sample
         level  - the trap signal (sum_k a_k s_k after instrument averaging) minus its record mean,
                  in units of the noise scale; drift and 1/f are NOT part of it, so y - level is
                  an estimate of the baseline + noise
         edge   - logit that a trap switched between t-1 and t
         step   - height of that switch (signed), noise-scale units
The trunk is the R7 dilated residual TCN (receptive field ~ 2^11 samples), heads are 1x1 convs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from net_r7 import Block


def normalise(y):
    """y: (B, T) -> x (B, 2, T), scale (B, 1).  Channels: first difference, level minus median."""
    d = torch.diff(y, dim=-1, prepend=y[..., :1])
    scale = d.abs().median(-1, keepdim=True).values / 0.6745 / np.sqrt(2) + 1e-6
    lvl = (y - y.median(-1, keepdim=True).values) / scale
    return torch.stack([d / scale, lvl], 1).clamp(-200, 200), scale


class StepNet(nn.Module):
    def __init__(self, C=64, n_dil=11):
        super().__init__()
        self.inp = nn.Conv1d(2, C, 5, padding=2)
        self.blocks = nn.Sequential(*[Block(C, 2 ** i) for i in range(n_dil)])
        self.head = nn.Sequential(nn.Conv1d(C, C, 1), nn.GELU(), nn.Conv1d(C, 3, 1))

    def forward(self, x):
        o = self.head(self.blocks(self.inp(x)))
        return dict(level=o[:, 0], edge=o[:, 1], step=o[:, 2])


@torch.no_grad()
def denoise(model, y, win=8192, overlap=1024, device="cpu"):
    """Long record -> trap-signal estimate in the record's own units (overlapping windows)."""
    y = torch.as_tensor(np.asarray(y, np.float32))
    T = y.numel()
    if T <= win:
        x, sc = normalise(y[None])
        o = model(x.to(device))
        return (o["level"][0].cpu() * sc[0]).numpy(), torch.sigmoid(o["edge"][0]).cpu().numpy(), (o["step"][0].cpu() * sc[0]).numpy()
    lev, edge, step = np.zeros(T), np.zeros(T), np.zeros(T)
    s = 0
    prev_end = 0
    while s < T:
        e = min(T, s + win)
        a = max(0, e - win)
        L, E, St = denoise(model, y[a:e].numpy(), win, overlap, device)
        keep0 = prev_end - a if a < prev_end else 0
        seg = slice(a + keep0, e)
        off = np.mean(lev[a:prev_end] - L[:keep0]) if keep0 > 0 else 0.0     # align the free offset
        lev[seg], edge[seg], step[seg] = L[keep0:] + off, E[keep0:], St[keep0:]
        prev_end = e
        if e == T:
            break
        s = e - overlap
    return lev, edge, step
