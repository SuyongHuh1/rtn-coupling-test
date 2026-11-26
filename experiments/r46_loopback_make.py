"""R46: generate the AO waveforms of the measurement-chain positive control (design: r46_loopback_design.md).

Two traps, coupled as in eq. (1) of the paper (phi = 0.5, symmetric J): when trap l is occupied, trap k's capture
rate is c_k exp(-J/2) and its emission rate e_k exp(J/2).  The joint 4-state chain is sampled exactly at the AO
sample instants (P = expm(Q)).  Base parameters per record: occupancy p ~ U[0.3, 0.7]; mean dwells in both
states log-uniform within 10-100 samples (rejection sampling).  Steps in whole AO LSB: the smaller trap step is
10 LSB, the other round(10 / r) LSB with r ~ U[0.15, 1]; which trap is smaller is random.  AO = sum of steps, 0 V
baseline.  Conditions: J = 0 (40), +1 (30), +2 (30), -1 (30); random playback order.

    python experiments/r46_loopback_make.py <out_dir>
-> <out_dir>/ao_in/NNN_<hash>.csv (5096 AO samples, volts) and results/r46/r46_truth.json (not used by the analysis)
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
SEED, N_AO, LSB, N_SMALL = 4646, 5096, 20 / 65536, 10
CONDS = [(0.0, 40), (1.0, 30), (2.0, 30), (-1.0, 30)]


def base_rates(rng):
    while True:
        p = rng.uniform(0.3, 0.7)
        t1 = np.exp(rng.uniform(np.log(10), np.log(100)))          # mean dwell occupied (emission time)
        t0 = t1 * (1 - p) / p                                       # mean dwell empty (capture time)
        if 10 <= t0 <= 100:
            return 1 / t0, 1 / t1, p, t0, t1                        # c, e


def generator(c, e, J):
    """Q over joint states s = n1 + 2 n2."""
    Q = np.zeros((4, 4))
    for s in range(4):
        n = (s & 1, s >> 1)
        for k in (0, 1):
            other = n[1 - k]
            ck, ek = c[k] * np.exp(-0.5 * J * other), e[k] * np.exp(0.5 * J * other)
            t = s ^ (1 << k)
            Q[s, t] = ek if n[k] else ck
    Q[np.diag_indices(4)] = -Q.sum(1)
    return Q


def stationary(Q):
    w, v = np.linalg.eig(Q.T)
    pi = np.real(v[:, np.argmin(np.abs(w))]); return pi / pi.sum()


def main():
    out = Path(sys.argv[1]); (out / "ao_in").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    recs = []
    for J, n in CONDS:
        for _ in range(n):
            c, e, prm = [], [], []
            for k in (0, 1):
                ck, ek, p, t0, t1 = base_rates(rng); c.append(ck); e.append(ek); prm.append(dict(p=p, tau0=t0, tau1=t1))
            r = rng.uniform(0.15, 1.0); big = int(round(N_SMALL / r)); small_first = bool(rng.random() < 0.5)
            steps = (N_SMALL, big) if small_first else (big, N_SMALL)
            P = expm(generator(c, e, J)); pi = stationary(generator(c, e, J))
            s = np.empty(N_AO, int); s[0] = rng.choice(4, p=pi)
            cum = np.cumsum(P, 1); u = rng.random(N_AO)
            for t in range(1, N_AO):
                s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), 3)
            n1, n2 = s & 1, s >> 1
            ao = (n1 * steps[0] + n2 * steps[1]) * LSB
            sw = [int(np.abs(np.diff(n1)).sum()), int(np.abs(np.diff(n2)).sum())]
            recs.append(dict(J=J, traps=prm, steps_LSB=list(steps), switches_ao=sw, ao=ao))
    order = rng.permutation(len(recs))
    truth = {}
    for pos, i in enumerate(order):
        r = recs[i]
        h = hashlib.sha1(f"{SEED}-{i}".encode()).hexdigest()[:8]
        name = f"{pos:03d}_{h}.csv"
        np.savetxt(out / "ao_in" / name, r.pop("ao"), fmt="%.9f")
        truth[name] = r
    (ROOT / "results/r46").mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=SEED, lsb_V=LSB, n_small_LSB=N_SMALL, records=truth), open(ROOT / "results/r46/r46_truth.json", "w"), indent=1)
    sw = np.array([min(v["switches_ao"]) for v in truth.values()])
    print(len(truth), "records; min switches per record: median", np.median(sw), "range", sw.min(), sw.max())


if __name__ == "__main__":
    main()
