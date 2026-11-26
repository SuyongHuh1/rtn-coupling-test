"""R46: analyse the measurement-chain loopback records (design: r46_loopback_design.md; no pipeline change).

Per AI record (volts, as recorded): the paper's two-trap calibrated test (r8_calib.job, --r9, B = 99) with
checklist steps 1-2 fields, J_hat; the four published rules on the same record (Guo |d| > 0.5, Albrecht e > 2 +
bootstrap rate check, Son chi2 p < 0.05, Zhang M_BIC < 2^K); and, for records called coupled, the pipeline
bootstrap interval for J (r24_jci.job, B = 39).  Truth (results/r46/r46_truth.json) is joined only in summary().
    RTN_MS=3 RTN_RESTART=3 RTN_JOINT=0 RTN_WORKERS=W python experiments/r46_loopback_analyze.py <ai_dir> [summary]
-> results/r46/r46_results.json (resumable)
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
AI = Path(sys.argv[1]) if len(sys.argv) > 1 else None
DO_SUMMARY = len(sys.argv) > 2 and sys.argv[2] == "summary"
sys.argv = [sys.argv[0], "full", "--r9"]                 # module-level argv parsing of the imported scripts
OUT = ROOT / "results/r46/r46_results.json"
B, B_JCI = 99, 39


def job(a):
    name, y = a
    import r8_calib as C, r10_baselines as R, r19_albrecht_faithful as A, r19c_son_faithful as S, r10b_zhang as Z
    R.COND = "full"; S.MODE = "full"
    i = int(name[:3])
    out = {}
    _, cal = C.job((i, y, 2, B))
    out.update(p=cal.get("p_r8"), stat=cal.get("stat_r8"), J=cal.get("J"), n_null=cal.get("n_null"),
               dwell_min=cal.get("dwell_min"), trans_min=cal.get("trans_min"), failed=cal.get("failed", False))
    for tag, f in (("guo", lambda: R.job((i, y, 2))[1]["pku_max"]),
                   ("albrecht", lambda: A.job((i, y, 2))[1]),
                   ("son_p", lambda: S.job((i, y, 2))[1]),
                   ("zhang_M", lambda: Z.job((i, y, 2))[1]["M_bic"])):
        try:
            out[tag] = f()
        except Exception as e:                                 # noqa: BLE001
            out[tag] = dict(error=repr(e)[:200])
    if out["p"] is not None and out["p"] <= 0.05:
        saved = sys.argv
        sys.argv = [saved[0], "full", "40", str(B_JCI)]      # r24_jci parses argv at import
        import r24_jci as JC
        sys.argv = saved
        JC.B = B_JCI
        _, jr = JC.job((i, y, float("nan")))
        out["jci"] = {k: v for k, v in jr.items() if k in ("J_hat", "Jb", "error")}
    return name, out


def summary():
    from math import sqrt
    def wil(k, n, z=1.96):
        if n == 0: return [None, None]
        p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(c - h, 3), round(c + h, 3)]
    res = json.load(open(OUT)); res = res.get("res", res); truth = json.load(open(ROOT / "results/r46/r46_truth.json"))["records"]
    ok12 = lambda r: (r.get("dwell_min") or 0) >= 3 and (r.get("trans_min") or 0) >= 15
    s = {}
    for J in (0.0, 1.0, 2.0, -1.0):
        names = [n for n, t in truth.items() if t["J"] == J and n in res and not res[n].get("failed")]
        rr = [res[n] for n in names]
        call = [r["p"] is not None and r["p"] <= 0.05 for r in rr]
        call12 = [c and ok12(r) for c, r in zip(call, rr)]
        row = dict(n=len(rr), called=int(sum(call)), rate=wil(int(sum(call)), len(rr)),
                   called_after_12=int(sum(call12)), pass12=int(sum(ok12(r) for r in rr)))
        if J != 0:
            Jh = [float(np.atleast_1d(r["J"])[0]) for r, c in zip(rr, call) if c]
            row["J_hat_median_called"] = float(np.median(Jh)) if Jh else None
            cov = []
            for r, c in zip(rr, call):
                if c and "jci" in r and "Jb" in r["jci"]:
                    Jb, Jh1 = np.array(r["jci"]["Jb"]), r["jci"]["J_hat"]; lo, hi = np.quantile(Jb, [0.025, 0.975])
                    cov.append(2 * Jh1 - hi <= J <= 2 * Jh1 - lo)
            row["coverage"] = [int(sum(cov)), len(cov)]
        rules = {}
        for tag, test in (("guo_d0.5", lambda r: isinstance(r.get("guo"), float) and r["guo"] > 0.5),
                          ("albrecht_e2_rate", lambda r: isinstance(r.get("albrecht"), dict) and r["albrecht"].get("e", 0) > 2 and r["albrecht"].get("rate_diff", False)),
                          ("son_p0.05", lambda r: isinstance(r.get("son_p"), float) and r["son_p"] < 0.05),
                          ("zhang_M<4", lambda r: isinstance(r.get("zhang_M"), int) and r["zhang_M"] < 4)):
            k = sum(test(r) for r in rr); rules[tag] = [int(k), len(rr)]
        row["published_rules"] = rules
        s[f"J={J:+g}"] = row
    s["failed"] = sum(1 for r in res.values() if isinstance(r, dict) and r.get("failed"))
    json.dump(dict(summary=s, res=res), open(OUT, "w"), indent=1)
    print(json.dumps(s, indent=1))


def main():
    files = sorted(AI.glob("[0-9][0-9][0-9]_*.csv"))
    res = json.load(open(OUT)).get("res", {}) if OUT.exists() else {}
    todo = [(f.name, np.loadtxt(f)) for f in files if f.name not in res]
    print(len(files), "records,", len(todo), "to do", flush=True)
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for name, r in p.imap_unordered(job, todo):
            res[name] = r
            json.dump(dict(res=res), open(OUT, "w"))
            print(name, "p", r["p"], "J", r["J"], flush=True)
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    summary() if DO_SUMMARY else main()
