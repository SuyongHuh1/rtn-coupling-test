"""R17f: pipeline-null calibration (39 nulls, instant) for the 3-source records that pass the checklist with
asymptotic p < 0.05 (asymptotic chi2 is anti-conservative for K = 3; records with asymptotic p >= 0.05 are
assumed not to become significant after calibration)."""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r17_rtninja_real import _job_b, DATA
from r17e_checklist import verdict


def main():
    NSRC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = json.load(open(ROOT / f"results/r8/r17e_checklist_{NSRC}src.json"))["rows"]
    recs = [r["rec"] for r in d if verdict(r, {}) == "coupling (false positive)"]
    base = DATA / f"n{NSRC}" / f"noise05_{NSRC}source"
    OUTP = ROOT / ("results/r8/r17f_cal3.json" if NSRC == 3 else f"results/r8/r17f_cal{NSRC}.json")
    out = json.load(open(OUTP)) if OUTP.exists() else []          # resume
    done = {r.get("rec") for r in out}
    todo = [(base / x, NSRC, 39, 6000 + i) for i, x in enumerate(recs) if x not in done and str(base / x) not in done]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as pool:
        for r in pool.imap_unordered(_job_b, todo):
            out.append(r); json.dump(out, open(OUTP, "w"), indent=1)
    print(json.dumps([(r["rec"], round(r.get("stat", -1), 1), r.get("p_cal"), r.get("error")) for r in out], indent=1))


if __name__ == "__main__":                # required on Windows (spawn)
    main()
