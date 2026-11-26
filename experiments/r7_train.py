"""R7: generate multi-trap training data, train the coupling network, evaluate on fresh data.

Runs on CUDA if available (recommended), else CPU.  Resumable: data chunks and per-epoch
checkpoints are written to --out and reused when the script is started again.

Usage (from the rtnsim folder):
    python experiments/r7_train.py                 # full run (GPU)
    python experiments/r7_train.py --quick         # small smoke test (CPU OK)

Outputs in --out (default results/r7): data/chunk_*.pt, ckpt.pt, model.pt (state_dict + config),
eval.json, train_log.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim.sim2_torch import make_batch, preprocess  # noqa: E402
from rtnsim.net_r7 import CouplingNet  # noqa: E402


def log(out, msg):
    print(msg, flush=True)
    with open(out / "train_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def generate(out, n_pairs, T, gen_batch, device, seed, tag):
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(n_pairs / gen_batch)
    paths = []
    for i in range(n_chunks):
        p = d / f"{tag}_{i:04d}.pt"
        paths.append(p)
        if p.exists():
            continue
        g = torch.Generator(device=device).manual_seed(seed + i)
        t0 = time.time()
        y, lab = make_batch(gen_batch, T, g, device)
        x = preprocess(y).half().cpu()
        torch.save(dict(x=x, coupled=lab["coupled"].cpu(), K=lab["K"].cpu().long(),
                        drive=lab["drive"].cpu()), p)
        log(out, f"[data {tag}] chunk {i + 1}/{n_chunks} K={int(lab['K'][0])} {time.time() - t0:.1f}s")
    return paths


def loss_fn(o, b):
    l_c = F.binary_cross_entropy_with_logits(o["coupled"], b["coupled"])
    pair = o["coupled"].view(-1, 2)                       # rows (coupled, matched independent)
    l_rank = F.softplus(-(pair[:, 0] - pair[:, 1])).mean()
    l_k = F.cross_entropy(o["K"], b["K"] - 2)
    l_d = F.binary_cross_entropy_with_logits(o["drive"], b["drive"])
    return l_c + 0.5 * l_rank + 0.3 * l_k + 0.3 * l_d


def batches(paths, bs, device, shuffle=True):
    order = torch.randperm(len(paths)) if shuffle else torch.arange(len(paths))
    for ci in order:
        ch = torch.load(paths[ci])
        n = ch["x"].shape[0] // 2
        perm = torch.randperm(n) if shuffle else torch.arange(n)
        for s in range(0, n, bs // 2):
            pi = perm[s:s + bs // 2]
            idx = torch.stack([2 * pi, 2 * pi + 1], 1).reshape(-1)        # keep pairs together
            yield {k: v[idx].to(device) for k, v in ch.items()}


@torch.no_grad()
def evaluate(model, paths, device, bs=256):
    model.eval()
    sc, lab, K, dr, kp = [], [], [], [], []
    for b in batches(paths, bs, device, shuffle=False):
        o = model(b["x"].float())
        sc.append(o["coupled"].cpu()); lab.append(b["coupled"].cpu()); K.append(b["K"].cpu())
        dr.append(b["drive"].cpu()); kp.append(o["K"].argmax(-1).cpu() + 2)
    sc, lab, K, dr, kp = map(torch.cat, (sc, lab, K, dr, kp))

    def roc_stats(s, y):
        s0, s1 = s[y == 0], s[y == 1]
        thr = torch.quantile(s0, 0.95)
        auc = float((s1[:, None] > s0[None]).float().mean()) if len(s1) * len(s0) < 4e8 else float("nan")
        return dict(auc=auc, tpr_at_fpr5=float((s1 > thr).float().mean()), n=int(len(s)))

    res = dict(all=roc_stats(sc, lab), K_accuracy=float((kp == K).float().mean()))
    for k in (2, 3, 4):
        m = K == k
        if m.any():
            res[f"K={k}"] = roc_stats(sc[m], lab[m])
    for name, m in (("with_drive", dr == 1), ("no_drive", dr == 0)):
        if m.any():
            res[name] = roc_stats(sc[m], lab[m])
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "r7"))
    ap.add_argument("--pairs", type=int, default=160000)
    ap.add_argument("--test-pairs", type=int, default=16000)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--gen-batch", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.pairs, a.test_pairs, a.T, a.gen_batch, a.epochs, a.bs, a.channels = 64, 32, 1024, 32, 1, 32, 16
        a.out = a.out + "_quick"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(out, f"device={device} torch={torch.__version__} args={vars(a)}")
    if device == "cuda":
        log(out, f"gpu={torch.cuda.get_device_name(0)}")
    train = generate(out, a.pairs, a.T, a.gen_batch, device, 1000, "train")
    test = generate(out, a.test_pairs, a.T, a.gen_batch, device, 900000, "test")
    model = CouplingNet(C=a.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = sum(1 for _ in range(0, a.pairs, a.bs // 2))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch + 10)
    start = 0
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start = st["epoch"] + 1
        log(out, f"resumed at epoch {start}")
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    for ep in range(start, a.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for b in batches(train, a.bs, device):
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                enabled=(device == "cuda")):
                o = model(b["x"].float())
                loss = loss_fn({k: v.float() for k, v in o.items()}, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); n += 1
        ev = evaluate(model, test, device)
        log(out, f"epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.4f} test {json.dumps(ev['all'])} "
                 f"K_acc {ev['K_accuracy']:.3f} ({time.time() - t0:.0f}s)")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep), ck)
    ev = evaluate(model, test, device)
    json.dump(ev, open(out / "eval.json", "w"), indent=1)
    torch.save(dict(state_dict=model.cpu().state_dict(), config=dict(C=a.channels, n_dil=10, c_in=2, T=a.T)),
               out / "model.pt")
    log(out, "done: " + json.dumps(ev))


if __name__ == "__main__":
    main()
