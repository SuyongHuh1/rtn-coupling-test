"""R3c: hardware path for the trained confound-aware causal CNN.

1. int8 quantisation (calibration on 400 synthetic training-distribution traces);
   float vs integer model on the R2 confound sets (FPR at 1% clean threshold, power)
   and on one efficiency condition.
2. Emit streaming HLS C++ with the trained weights; C-simulate with g++ and check
   bit-exactness against the integer reference on held-out traces.
3. Analytical cost: MACs and energy per sample (Horowitz 45 nm) for the CNN (int8 / fp32)
   and the HMM-EM GLRT (fp32, multi-pass).

Outputs results/r3/r3_hw.json and results/r3/hls/ (sources + test vectors).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim import learned_causal as LC  # noqa: E402
from rtnsim import quant as Q  # noqa: E402
from rtnsim import hls_gen as H  # noqa: E402
from rtnsim import theory as th  # noqa: E402
import r2_detectors as R2  # noqa: E402
import r3_cnn as R3  # noqa: E402
from r1_bound import base  # noqa: E402

OUT = ROOT / "results" / "r3"


def em_cost_per_sample(passes=140):
    """4-state forward-backward with E-step counts: ~16 MAC forward, ~32 MAC backward incl.
    transition counts, ~16 MAC for counts, 8 exp, 8 div per sample per pass."""
    mac = 64
    exps = 8
    e_pass = mac * (3.7 + 0.9) + exps * 10 * (3.7 + 0.9)    # exp ~ 10 flop-equivalents
    return dict(passes=passes, flop_equiv_per_sample=passes * (mac * 2 + exps * 20),
                energy_J=passes * e_pass * 1e-12)


def main(mode="level"):
    LC.INPUT = mode
    tag = "" if mode == "level" else "_diff"
    p = LC.load(OUT / f"ccnn{tag}_params.pkl")
    yc, _ = LC.make_dataset(200, 1024, np.random.default_rng(123))
    q = Q.quantize(p, yc)
    res = {}
    fl = R3.eval_confounds(lambda y: LC.scores(p, y))
    it = R3.eval_confounds(lambda y: Q.scores_int(q, y))
    res["confounds_float"], res["confounds_int8"] = fl, it

    # held-out bit-exactness of the emitted C++
    d = H.emit(q, OUT / f"hls{tag}")
    yt, _ = LC.make_dataset(4, 2000, np.random.default_rng(321))
    xq = Q.quantize_input(q, yt)
    (d / "in.txt").write_text(f"{xq.shape[0]} {xq.shape[1]}\n" + " ".join(map(str, xq.ravel())))
    r = subprocess.run(f"g++ -O2 -std=c++17 -Wno-unknown-pragmas -o {d}/tb {d}/tb.cpp {d}/ccnn.cpp && "
                       f"{d}/tb {d}/in.txt {d}/out.txt", shell=True, capture_output=True, text=True)
    out = np.loadtxt(d / "out.txt", dtype=np.int64)
    S1, S2 = Q.run_int(q, xq)
    res["csim_bit_exact"] = bool(r.returncode == 0 and np.array_equal(out[:, :32], S1) and np.array_equal(out[:, 32:], S2))
    np.savetxt(d / "expected_S1S2.txt", np.concatenate([S1, S2], axis=1), fmt="%d")

    cm = Q.cost_model(fs_hz=1e5)
    res["cost"] = dict(cnn=cm, em_glrt=em_cost_per_sample(), params=LC.n_params(p),
                       receptive_field=LC.receptive_field(),
                       weight_bytes_int8=int(sum(l["w"].size for l in q["layers"])),
                       line_buffer_bytes=int(len(q["layers"]) * ((LC.KR - 1) * max(LC.DIL) + 1) * LC.CH))
    res["input"] = mode
    json.dump(res, open(OUT / f"r3_hw{tag}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "confounds_float" and k != "confounds_int8"}, indent=1,
                     default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "level")
