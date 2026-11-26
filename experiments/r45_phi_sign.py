"""R45 (review): does fixing phi = 0.5 (and symmetric J) bias detection?  Power of the calibrated three-trap
test on the R44 coupled set (100 records, 1/f) split by the simulator's true phi (drawn in [0.2, 0.8]) and by
the sign of the coupling.  phi is not stored in the labels, so the R44 records are regenerated with make_batch
patched to also return phi (same random stream; the records are checked to be identical).
    python experiments/r45_phi_sign.py  -> results/r8/r45_phi_sign.json
"""
import inspect, json, sys, textwrap
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from rtnsim import sim2_torch as S
import r40_replication as R

src = textwrap.dedent(inspect.getsource(S.make_batch))
a = "    labels = dict(coupled="
assert src.count(a) == 1
ns = dict(vars(S)); exec(src.replace(a, "    labels = dict(phi=torch.cat([phi, phi]).to(dev), coupled=").replace("def make_batch", "def mb_phi"), ns)
recs = R.make_pairs()
g = torch.Generator().manual_seed(R.SEED + 3); got, idx, phi, J = 0, 200, {}, {}
while got < R.N:
    y, lab = ns["mb_phi"](50, R.T, g, "cpu", p_drive=0, p_box=0, return_states=True)
    if int(lab["K"][0]) != 3:
        continue
    take = min(50, R.N - got)
    for r in range(2 * take):
        assert np.allclose(y[r].double().numpy(), recs[idx]["y"])
        phi[idx] = float(lab["phi"][r]); J[idx] = lab["Jvec"][r].tolist(); idx += 1
    got += take
res = json.load(open(ROOT / "results/r8/r44_rep_power.json"))
rows = [dict(k=k, phi=phi[int(k)], Jtop=max(J[int(k)], key=abs), sig=v["orig"]["sig"]) for k, v in res.items() if k != "summary"]
def pw(sel): s = [r["sig"] for r in rows if sel(r)]; return [sum(s), len(s)]
out = dict(rows=rows, by_phi={"|phi-0.5|<0.1": pw(lambda r: abs(r["phi"] - .5) < .1), "0.1-0.2": pw(lambda r: .1 <= abs(r["phi"] - .5) < .2),
                               ">=0.2": pw(lambda r: abs(r["phi"] - .5) >= .2)},
           by_sign={"repulsive": pw(lambda r: r["Jtop"] > 0), "attractive": pw(lambda r: r["Jtop"] < 0)})
json.dump(out, open(ROOT / "results/r8/r45_phi_sign.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("by_phi", "by_sign")}))
