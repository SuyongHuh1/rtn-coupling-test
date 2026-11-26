"""Verilog RTL package for the int8 streaming causal CNN (DE10-Nano / Cyclone V, Quartus).

emit(q, out_dir, xq=None) writes
  ccnn_core.v            the streaming engine (copied from rtl/)
  ccnn_params.vh         Q_SHIFT
  ccnn_w.hex             704 x 256-bit weight words, address = rombase(l) + o*K + k,
                         byte c (bits 8c+7..8c) = W_l[k][c][o]; layer 0: byte k = W0[k][0][o]
  ccnn_b.hex / ccnn_m.hex / ccnn_ms.hex   256 x 32-bit (192 used) bias / multiplier / skip multiplier (l*32+o)
and, when integer test traces xq (B, T) are given,
  tv_in.hex              B*T signed int8 inputs (two's complement)
  tv_exp.hex             per trace: S1[0..31] then S2[0..31], 48-bit
  tb_core.v              iverilog testbench (feeds all traces, compares, prints PASS/FAIL)
  de10_selftest.v        on-board self-test top (same vectors, LEDs)
  create_project.tcl     Quartus project for 5CSEBA6U23I7 with DE10-Nano pin assignments
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import quant as Q

RTL_SRC = Path(__file__).resolve().parents[1] / "rtl"
ROMBASE = [0, 32, 192, 352, 512, 672]
NCH = 32


def _word(bytes_):
    """list of 32 int8 -> 64-hex-digit string, byte 0 in the least significant position."""
    return "".join(f"{int(b) & 0xFF:02x}" for b in reversed(bytes_))


def weight_words(q):
    words = []
    for l, L in enumerate(q["layers"]):
        w = L["w"]
        K, Cin, Cout = w.shape
        assert Cout == NCH and len(words) == ROMBASE[l]
        for o in range(Cout):
            if l == 0:
                words.append(_word([w[k, 0, o] for k in range(K)] + [0] * (NCH - K)))
            else:
                for k in range(K):
                    words.append(_word([w[k, c, o] for c in range(Cin)]))
    assert len(words) == 704
    return words


def _hex32(v):
    v = int(v)
    assert -(1 << 31) <= v < (1 << 31)
    return f"{v & 0xFFFFFFFF:08x}"


def emit(q, out_dir, xq=None, S1=None, S2=None):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    assert all(L["shift"] == 24 for L in q["layers"])
    shutil.copy(RTL_SRC / "ccnn_core.v", d / "ccnn_core.v")
    (d / "ccnn_params.vh").write_text("`define Q_SHIFT 24\n")
    (d / "ccnn_w.hex").write_text("\n".join(weight_words(q)) + "\n")
    b, m, ms = [], [], []
    for L in q["layers"]:
        b += list(L["b"])
        m += list(L["m"])
        ms += list(L.get("m_skip", np.zeros(NCH, np.int64)))
    pad = [0] * (256 - len(b))          # memories are 256 deep ({l, o} address); unused tail = 0
    b, m, ms = b + pad, m + pad, ms + pad
    for name, arr in (("ccnn_b.hex", b), ("ccnn_m.hex", m), ("ccnn_ms.hex", ms)):
        (d / name).write_text("\n".join(_hex32(v) for v in arr) + "\n")
    if xq is not None:
        B, T = xq.shape
        if S1 is None:
            S1, S2 = Q.run_int(q, xq)
        (d / "tv_in.hex").write_text("\n".join(f"{int(v) & 0xFF:02x}" for v in xq.ravel()) + "\n")
        exp = []
        for i in range(B):
            exp += [f"{int(v):012x}" for v in S1[i]] + [f"{int(v):012x}" for v in S2[i]]
        (d / "tv_exp.hex").write_text("\n".join(exp) + "\n")
        for name in ("tb_core.v", "de10_selftest.v", "create_project.tcl"):
            txt = (RTL_SRC / name).read_text().replace("@NTR@", str(B)).replace("@NT@", str(T))
            (d / name).write_text(txt)
    return d
