"""Loaders for public RTN datasets (Keysight B1500A CSV exports)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_b1500(path, current="Drain_I"):
    """Return (t, I, meta) from a B1500A EasyEXPERT CSV export."""
    names, rows, meta = None, [], {}
    for line in Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if line.startswith("DataName"):
            names = [s.strip() for s in line.split(",")[1:]]
        elif line.startswith("DataValue"):
            rows.append([float(s) for s in line.split(",")[1:]])
        elif line.startswith("TestParameter, Channel.IName"):
            meta["inames"] = [s.strip() for s in line.split(",")[2:]]
        elif line.startswith("TestParameter, Channel.VName"):
            meta["vnames"] = [s.strip() for s in line.split(",")[2:]]
    a = np.array(rows)
    if names is None:                       # EDTM files carry no DataName line: V..., Time, I...
        names = meta["vnames"] + ["Time"] + meta["inames"]
        current = {"Drain_I": "Idrain"}.get(current, current)
    if current not in names and current == "Drain_I":
        current = "Idrain"
    meta["names"] = names
    return a[:, names.index("Time")], a[:, names.index(current)], meta
