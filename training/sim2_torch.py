"""Batched multi-trap RTN simulator in PyTorch (CPU or CUDA) for training the R7 network.

Each record: K in {1, 2, 3, 4} two-state traps (time measured in samples after rescaling),
optional kinetic couplings J_ij, optional slow shared telegraph drive, additive levels with
possibly near-degenerate steps, white + 1/f-like noise, random-walk drift, and optional
boxcar averaging (instrument moving average, simulated by 4x oversampling).

Matched pairs (as in R3): every coupled record has an independent partner with the SAME
nuisance draw (K, steps, noise, drift, drive, averaging) whose trap rates are the I-projection
of the coupled chain (each trap keeps its occupancy and switching flux), so the only
difference the network can use is the joint (coupling) statistics.

Exact discrete-time sampling: generator Q (2^K x 2 states), P = expm(Q dt) per record.
"""
from __future__ import annotations

import math

import torch

KMAX = 4


def _bits(K, device):
    x = torch.arange(2 ** K, device=device)
    return ((x[:, None] >> torch.arange(K, device=device)[None]) & 1).float()      # (S, K)


def generator(c, e, J, phi, gamma=None, drive_rates=None):
    """c, e: (B, K); J: (B, K, K) symmetric zero-diagonal; phi: (B,);
    gamma: (B, K) drive shifts or None; drive_rates: (B, 2) [0->1, 1->0] or None.
    Returns Q: (B, S, S) with S = 2^K (x2 with drive), state index = sum s_k 2^k + 2^K u."""
    B, K = c.shape
    dev = c.device
    bits = _bits(K, dev).to(c.dtype)                       # (S0, K)
    S0 = 2 ** K
    nu = 1 if gamma is None else 2
    S = S0 * nu
    Q = torch.zeros(B, S, S, device=dev, dtype=c.dtype)
    for u in range(nu):
        h = torch.einsum("bkj,sj->bsk", J, bits)          # (B, S0, K) field from other traps
        if gamma is not None and u == 1:
            h = h + gamma[:, None, :]
        cap = c[:, None, :] * torch.exp(-phi[:, None, None] * h)
        emi = e[:, None, :] * torch.exp((1 - phi[:, None, None]) * h)
        rate = torch.where(bits[None] == 0, cap, emi)      # (B, S0, K) flip rate of trap k
        for k in range(K):
            src = torch.arange(S0, device=dev) + u * S0
            dst = (torch.arange(S0, device=dev) ^ (1 << k)) + u * S0
            Q[:, src, dst] = rate[:, :, k]
    if gamma is not None:
        idx = torch.arange(S0, device=dev)
        Q[:, idx, idx + S0] = drive_rates[:, 0:1].expand(B, S0)
        Q[:, idx + S0, idx] = drive_rates[:, 1:2].expand(B, S0)
    Q = Q - torch.diag_embed(Q.sum(-1))
    return Q


def stationary(Q):
    B, S, _ = Q.shape
    A = torch.cat([Q.transpose(1, 2), torch.ones(B, 1, S, device=Q.device)], 1)
    b = torch.zeros(B, S + 1, 1, device=Q.device)
    b[:, -1] = 1
    pi = torch.linalg.lstsq(A, b).solution[..., 0]
    return pi.clamp_min(0) / pi.clamp_min(0).sum(-1, keepdim=True)


def i_projection(c, e, J, phi):
    """Rates of the closest independent chain (no drive): conditional flip rates averaged over
    the stationary distribution of the coupled chain."""
    B, K = c.shape
    Q = generator(c, e, J, phi)
    pi = stationary(Q)
    bits = _bits(K, c.device)
    S0 = 2 ** K
    cs, es = [], []
    for k in range(K):
        dst = torch.arange(S0, device=c.device) ^ (1 << k)
        r = Q[:, torch.arange(S0, device=c.device), dst]   # (B, S0)
        m0, m1 = (bits[:, k] == 0).float(), (bits[:, k] == 1).float()
        cs.append((pi * r * m0).sum(-1) / (pi * m0).sum(-1).clamp_min(1e-12))
        es.append((pi * r * m1).sum(-1) / (pi * m1).sum(-1).clamp_min(1e-12))
    return torch.stack(cs, -1), torch.stack(es, -1)


def sample_chain(Q, T, dt, gen):
    """Exact discrete-time samples of state indices, (B, T) long."""
    P = torch.linalg.matrix_exp(Q * dt).clamp_min(0)
    P = P / P.sum(-1, keepdim=True)
    C = P.cumsum(-1)
    C[..., -1] = 1.0 + 1e-6
    pi = stationary(Q)
    B, S, _ = Q.shape
    x = torch.multinomial(pi.clamp_min(1e-12), 1, generator=gen)[:, 0]
    out = torch.empty(B, T, dtype=torch.long, device=Q.device)
    u = torch.rand(T, B, 1, device=Q.device, generator=gen)
    ar = torch.arange(B, device=Q.device)
    for t in range(T):
        x = torch.searchsorted(C[ar, x], u[t]).squeeze(-1).clamp_max(S - 1)
        out[:, t] = x
    return out


def flicker(B, T, amp, gen):
    """1/f noise by spectral shaping of white noise (f >= 1/T), unit variance times amp."""
    dev = amp.device
    w = torch.randn(B, T, device=dev, generator=gen)
    W = torch.fft.rfft(w, dim=-1)
    f = torch.fft.rfftfreq(T, device=dev)
    shape = torch.where(f > 0, 1.0 / torch.sqrt(f.clamp_min(1.0 / T)), torch.zeros_like(f))
    x = torch.fft.irfft(W * shape, n=T, dim=-1)
    x = x / x.std(-1, keepdim=True).clamp_min(1e-9)
    return amp[:, None] * x


def make_batch(B, T, gen, device, p_drive=0.3, p_drift=0.5, p_box=0.5, oversample=4, J_range=(0.7, 3.0),
               return_states=False, p_flicker=1.0, return_clean=False):
    """Returns y (2B, T) float, labels dict: coupled (2B,), K (2B,), drive (2B,).
    Rows [2i, 2i+1] = (coupled, matched independent) with identical nuisance."""
    dev = device
    K = int(torch.randint(2, KMAX + 1, (1,), generator=gen, device=gen.device))   # one K per batch (fixed shapes)
    U = lambda *s: torch.rand(*s, device=dev, generator=gen)
    tau = torch.exp(math.log(8) + U(B, K) * (math.log(200) - math.log(8)))        # mean dwell scale (samples)
    p = 0.15 + 0.7 * U(B, K)
    k_tot = 2.0 / tau
    c, e = k_tot * p, k_tot * (1 - p)
    J = torch.zeros(B, K, K, device=dev)
    i = torch.randint(0, K, (B,), device=dev, generator=gen)
    j = (i + 1 + torch.randint(0, K - 1, (B,), device=dev, generator=gen)) % K
    Jv = (J_range[0] + (J_range[1] - J_range[0]) * U(B)) * torch.where(U(B) < 0.5, -1.0, 1.0)
    J[torch.arange(B), i, j] = Jv
    J[torch.arange(B), j, i] = Jv
    phi = 0.2 + 0.6 * U(B)
    c0, e0 = i_projection(c, e, J, phi)
    drive = U(B) < p_drive
    gamma = torch.randn(B, K, device=dev, generator=gen) * 1.5 * drive[:, None]
    dwell_u = tau.max(-1).values * (10 + 40 * U(B))
    drate = torch.stack([1 / dwell_u, 1 / dwell_u], -1)
    dt = 1.0 / oversample
    Tf = T * oversample
    Q1 = generator(c, e, J, phi, gamma, drate)
    Q0 = generator(c0, e0, torch.zeros_like(J), phi, gamma, drate)
    x = sample_chain(torch.cat([Q1, Q0]), Tf, dt, gen)                             # (2B, Tf)
    bits = _bits(K, dev)
    s = bits[x % (2 ** K)]                                                         # (2B, Tf, K)
    a = torch.sort(0.15 + 0.85 * U(B, K), -1, descending=True).values
    a[:, 0] = 1.0
    a = torch.cat([a, a])
    lev = (s * a[:, None, :]).sum(-1)
    box = torch.cat([U(B) < p_box] * 2)
    lev_box = lev.view(2 * B, T, oversample).mean(-1)
    lev_pt = lev[:, ::oversample]
    y = torch.where(box[:, None], lev_box, lev_pt)
    y_clean = y.clone()
    smallest = a.min(-1).values
    sig = smallest * (0.15 + 0.6 * torch.cat([U(B)] * 2))
    gw = torch.Generator(device=dev).manual_seed(int(torch.randint(0, 2 ** 31, (1,), generator=gen, device=gen.device)))
    noise = torch.randn(B, T, device=dev, generator=gw)
    y = y + sig[:, None] * torch.cat([noise, noise])                               # same noise realisation in the pair
    fl_amp = torch.cat([0.6 * sig[:B] * U(B) * (U(B) < p_flicker)] * 2)
    fl = flicker(B, T, fl_amp[:B], gw)
    y = y + torch.cat([fl, fl])
    drift_on = torch.cat([U(B) < p_drift] * 2)
    dw = torch.randn(B, T, device=dev, generator=gw).cumsum(-1) / math.sqrt(T)
    y = y + (drift_on * torch.cat([U(B)] * 2))[:, None] * torch.cat([dw, dw]) * a[:, :1]
    # parameter targets (traps ordered by step size, padded to KMAX): log total rate, logit occupancy, step
    cc, ee = torch.cat([c, c0]), torch.cat([e, e0])
    th_ = torch.zeros(2 * B, KMAX, 3, device=dev)
    th_[:, :K, 0] = torch.log(cc + ee)
    th_[:, :K, 1] = torch.log(cc / ee)
    th_[:, :K, 2] = a
    mask = torch.zeros(2 * B, KMAX, device=dev)
    mask[:, :K] = 1
    iu = torch.triu_indices(KMAX, KMAX, 1, device=dev)
    Jp = torch.zeros(2 * B, KMAX, KMAX, device=dev)
    Jp[:B, :K, :K] = J
    labels = dict(coupled=torch.cat([torch.ones(B), torch.zeros(B)]).to(dev),
                  K=torch.full((2 * B,), K, device=dev), drive=torch.cat([drive, drive]).float(),
                  theta=th_, theta_mask=mask, Jvec=Jp[:, iu[0], iu[1]],
                  Jmask=(mask[:, iu[0]] * mask[:, iu[1]]))
    order = torch.stack([torch.arange(B), torch.arange(B) + B], 1).reshape(-1).to(dev)
    if return_clean:
        labels["clean"] = y_clean                                                    # trap signal only (after instrument averaging)
        labels["smallest_step"] = a.min(-1).values
    if return_states:
        labels["states"] = s[:, ::oversample].to(torch.int8)                          # (2B, T, K), point samples
        labels["steps"] = a
    return y[order], {k: v[order] for k, v in labels.items()}


def preprocess(y):
    """Per-record robust standardisation; channels = [first difference, level minus slow mean]."""
    d = torch.diff(y, dim=-1, prepend=y[..., :1])
    scale = d.abs().median(-1, keepdim=True).values / 0.6745 + 1e-6
    W = min(1025, y.shape[-1] // 2 * 2 + 1)
    pad = W // 2
    yp = torch.nn.functional.pad(y[:, None], (pad, pad), mode="replicate")
    slow = torch.nn.functional.avg_pool1d(yp, W, stride=1)[:, 0]
    lvl = (y - slow) / scale
    return torch.stack([d / scale, lvl], 1).clamp(-50, 50)
