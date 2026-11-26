"""R19c: Son, Kim, Kang IEICE ELEX 2017 (doi 10.1587/elex.14.20161189) read in full.

Full text: the larger trap (trap1) is quantised and subtracted; trap2's capture/emission times are then
extracted (original method) from the whole record, (proposed) only while trap1 is NOT captured.  Dependence
is judged by whether the two methods give different tau_c/tau_e (and hence trap depth/energy) beyond the
margin of error; there is no statistical test, threshold or Bonferroni, and only ONE direction (trap2 | trap1)
is examined, for 4-level (2-trap) RTN.
Single-record formalisation closest to the text: on the idealised path, trap2 (smaller step) rates while
trap1 empty vs while trap1 full, LRT chi2(2), one test at 0.05 (no multiplicity correction).  For K = 3
the same test is applied to (smallest | largest) only.
    python experiments/r19c_son_faithful.py white|full|realneg
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "white"
import r8_oracle as O
from rtnsim.robustk import idealise_k
from rtnsim.surrogate import _ll_binary_transitions


def son_p(S):
    t2, t1 = S[:, -1], S[:, 0]
    g = _ll_binary_transitions(t2[:-1], t2[1:], t1[:-1]) - _ll_binary_transitions(t2[:-1], t2[1:])
    return float(chi2.sf(2 * max(g, 0), 2))


def job(a):
    i, y, K = a
    f = idealise_k(y, K, baseline=(MODE != "white"), rng=np.random.default_rng(i))
    return i, son_p(f["S"])


if __name__ == "__main__":
    if MODE == "realneg":
        sys.argv = [sys.argv[0]]
        from r12_realneg import records
        ys = records(40)
        with Pool(2) as p:
            out = dict(p.map(job, [(i, y, 2) for i, y in enumerate(ys)]))
        s = dict(n=40, son_faithful_fpr=float(np.mean([v < 0.05 for v in out.values()])))
    else:
        ys, ss, rows = O.make(100, white=(MODE == "white"))
        with Pool(2) as p:
            out = dict(p.map(job, [(i, ys[i], rows[i]["K"]) for i in range(len(ys))], chunksize=4))
        s = {}
        for K in (2, 3):
            idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
            lab = np.array([rows[i]["coupled"] for i in idx]); c = np.array([out[i] < 0.05 for i in idx])
            s[f"K={K}"] = dict(fpr=float(c[lab == 0].mean()), tpr=float(c[lab == 1].mean()))
    json.dump(dict(summary=s, p={str(k): v for k, v in out.items()}), open(ROOT / f"results/r8/r19c_son_{MODE}.json", "w"), indent=1)
    print(MODE, json.dumps(s))
