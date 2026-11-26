"""R9a: train the per-sample step detector on the GPU (data generated on the fly).

    python train_step.py --quick     # 1-minute check
    python train_step.py             # full run (~45-60 min on an RTX 5060 Ti)
Resumable.  Results in ./out/ (stepnet.pt, eval.json, train_log.txt).  Only PyTorch needed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sim2_torch import make_batch
from stepnet import StepNet, normalise

HERE = Path(__file__).resolve().parent


def log(out, msg):
    print(msg, flush=True)
    with open(out / "train_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def gen_chunk(n_pairs, T, seed, device, parts=4):
    """Mixed-K chunk: several simulator calls (each draws its own K), concatenated and shuffled."""
    xs, lv, ed, st, Ks = [], [], [], [], []
    for p in range(parts):
        g = torch.Generator(device=device).manual_seed(seed * 16 + p)
        y, lab = make_batch(max(1, n_pairs // parts), T, g, device, return_clean=True)
        x, scale = normalise(y)
        clean = lab["clean"]
        d = torch.diff(clean, dim=-1, prepend=clean[..., :1])
        xs.append(x)
        lv.append((clean - clean.mean(-1, keepdim=True)) / scale)
        ed.append((d.abs() > 0.25 * lab["smallest_step"][:, None]).float())
        st.append(d / scale)
        Ks.append(lab["K"])
    cat = lambda z: torch.cat(z)
    x, lv, ed, st, K = map(cat, (xs, lv, ed, st, Ks))
    perm = torch.randperm(x.shape[0], device=device)
    return x[perm], lv[perm], ed[perm], st[perm], K[perm]


def loss_fn(o, lv, ed, st):
    l_lv = F.huber_loss(o["level"] - o["level"].mean(-1, keepdim=True), lv, delta=1.0)
    l_ed = F.binary_cross_entropy_with_logits(o["edge"], ed, pos_weight=torch.tensor(10.0, device=ed.device))
    m = ed > 0
    l_st = F.huber_loss(o["step"][m], st[m], delta=1.0) if m.any() else o["step"].sum() * 0
    return l_lv + l_ed + 0.5 * l_st, (l_lv.item(), l_ed.item(), l_st.item())


@torch.no_grad()
def evaluate(model, test, bs=64):
    model.eval()
    tot = dict(level_rmse=0.0, edge_tp=0, edge_fp=0, edge_fn=0, step_mae=0.0, n=0, ns=0)
    for x, lv, ed, st, K in test:
        for s in range(0, x.shape[0], bs):
            o = model(x[s:s + bs].float())
            L = o["level"] - o["level"].mean(-1, keepdim=True)
            tot["level_rmse"] += float(((L - lv[s:s + bs]) ** 2).mean(-1).sqrt().sum())
            tot["n"] += L.shape[0]
            pe = (torch.sigmoid(o["edge"]) > 0.5).float()
            te = ed[s:s + bs]
            # tolerance +-2 samples
            tdil = F.max_pool1d(te[:, None], 5, 1, 2)[:, 0]
            pdil = F.max_pool1d(pe[:, None], 5, 1, 2)[:, 0]
            tot["edge_tp"] += float((pe * tdil).sum()); tot["edge_fp"] += float((pe * (1 - tdil)).sum())
            tot["edge_fn"] += float((te * (1 - pdil)).sum())
            m = te > 0
            tot["step_mae"] += float((o["step"][m] - st[s:s + bs][m]).abs().sum()); tot["ns"] += int(m.sum())
    model.train()
    p = tot["edge_tp"] / max(tot["edge_tp"] + tot["edge_fp"], 1)
    r = tot["edge_tp"] / max(tot["edge_tp"] + tot["edge_fn"], 1)
    return dict(level_rmse=tot["level_rmse"] / tot["n"], edge_precision=p, edge_recall=r,
                edge_f1=2 * p * r / max(p + r, 1e-9), step_mae=tot["step_mae"] / max(tot["ns"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256, help="pairs per generated chunk")
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64, help="records per step")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = HERE / ("out_quick" if a.quick else "out")
    if a.quick:
        a.steps, a.T, a.chunk, a.bs, a.channels, a.eval_every = 60, 1024, 32, 16, 16, 30
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(out, f"device={device} torch={torch.__version__} " + (torch.cuda.get_device_name(0) if device == "cuda" else "") + f" args={vars(a)}")
    torch.backends.cudnn.benchmark = True
    log(out, "generating test set ...")
    test = [gen_chunk(a.chunk, a.T, 9_000_000 + i, device) for i in range(2 if a.quick else 8)]
    model = StepNet(C=a.channels).to(device)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(1000, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, ck = 0, out / "ckpt.pt"
    if ck.exists():
        stt = torch.load(ck, map_location=device)
        model.load_state_dict(stt["model"]); ema.load_state_dict(stt["ema"])
        opt.load_state_dict(stt["opt"]); sched.load_state_dict(stt["sched"]); step = stt["step"]
        log(out, f"resumed at step {step}")
    t0, run = time.time(), None
    seed = 1 + step * 131
    while step < a.steps:
        x, lv, ed, st, K = gen_chunk(a.chunk, a.T, seed, device)
        seed += 1
        n = x.shape[0]
        idx = torch.cat([torch.randperm(n, device=device) for _ in range(a.reuse)])
        for s in range(0, len(idx), a.bs):
            b = idx[s:s + a.bs]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                o = model(x[b])
            loss, parts = loss_fn({k: v.float() for k, v in o.items()}, lv[b], ed[b], st[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            step += 1
            if step % a.eval_every == 0 or step >= a.steps:
                ev = evaluate(ema, test)
                log(out, f"step {step}/{a.steps} loss {run:.4f} ema-test {json.dumps({k: round(v, 4) for k, v in ev.items()})} ({time.time() - t0:.0f}s)")
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                                sched=sched.state_dict(), step=step), ck)
            if step >= a.steps:
                break
    ev = evaluate(ema, test)
    json.dump(ev, open(out / "eval.json", "w"), indent=1)
    torch.save(dict(state_dict={k: v.cpu() for k, v in ema.state_dict().items()}, config=dict(C=a.channels, n_dil=11)),
               out / "stepnet.pt")
    log(out, "done: " + json.dumps(ev))


if __name__ == "__main__":
    main()
