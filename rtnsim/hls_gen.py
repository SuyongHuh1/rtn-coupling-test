"""R3c: emit a streaming C++ (Vitis HLS style) implementation of the int8 causal CNN.

Files written to out_dir:
  ccnn_weights.h  constants, int8 weights, int32 biases and requantisation multipliers
  ccnn.h / ccnn.cpp  one-sample-per-call kernel `ccnn_step` (line buffers per layer,
                    integer MACs, ReLU, requantisation, residual adds, running sums)
  tb.cpp          C-simulation testbench: reads int8 input traces, writes S1/S2
The kernel is plain C++17 (HLS pragmas are ignored by g++), so the same source
runs as a bit-exact C simulation here and as the HLS input on the FPGA tool flow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KERNEL_H = r"""#pragma once
#include <cstdint>
#include "ccnn_weights.h"

struct CcnnState {
    int8_t  buf[N_LAYERS][MAX_DEPTH][MAX_C];   // causal line buffers (circular)
    int     head[N_LAYERS];
    int64_t S1[C_FEAT];
    int64_t S2[C_FEAT];
    int64_t n;
};

void ccnn_reset(CcnnState &st);
// One ADC sample (already normalised and quantised to int8) per call.
void ccnn_step(int8_t x, CcnnState &st);
"""

KERNEL_CPP = r"""#include "ccnn.h"

static inline int32_t clamp8(int64_t v) { return v < 0 ? 0 : (v > 127 ? 127 : (int32_t)v); }
static inline int64_t requant(int64_t acc, int64_t m, int shift) {
    return (acc * m + ((int64_t)1 << (shift - 1))) >> shift;
}

void ccnn_reset(CcnnState &st) {
    for (int l = 0; l < N_LAYERS; ++l) {
        st.head[l] = 0;
        for (int d = 0; d < MAX_DEPTH; ++d)
            for (int c = 0; c < MAX_C; ++c) st.buf[l][d][c] = 0;
    }
    for (int c = 0; c < C_FEAT; ++c) { st.S1[c] = 0; st.S2[c] = 0; }
    st.n = 0;
}

// Generic causal dilated conv layer with optional residual.
template <int K, int DIL, int CIN, int COUT, bool RES>
static void layer(int l, const int8_t in[CIN], int8_t out[COUT], CcnnState &st,
                  const int8_t W[K][CIN][COUT], const int32_t B[COUT],
                  const int32_t M[COUT], const int32_t MSKIP[COUT], int shift) {
#pragma HLS INLINE
    constexpr int DEPTH = (K - 1) * DIL + 1;
    int h = st.head[l];
    for (int c = 0; c < CIN; ++c) st.buf[l][h][c] = in[c];       // newest sample at h
    for (int o = 0; o < COUT; ++o) {
#pragma HLS PIPELINE II=1
        int64_t acc = B[o];
        for (int k = 0; k < K; ++k) {
            int back = (K - 1 - k) * DIL;                         // x[t - back]
            int idx = h - back; if (idx < 0) idx += DEPTH;
            for (int c = 0; c < CIN; ++c) acc += (int64_t)st.buf[l][idx][c] * W[k][c][o];
        }
        int64_t r = clamp8(requant(acc > 0 ? acc : 0, M[o], shift));
        if (RES) {
            int64_t s = clamp8(requant(in[o], MSKIP[o], shift));
            int64_t v = s + r; out[o] = (int8_t)(v > 127 ? 127 : v);
        } else {
            out[o] = (int8_t)r;
        }
    }
    st.head[l] = (h + 1) % DEPTH;
}

void ccnn_step(int8_t x, CcnnState &st) {
#pragma HLS INTERFACE ap_none port=x
    int8_t a0[1] = {x};
    int8_t a1[NCH], a2[NCH], a3[NCH], a4[NCH], a5[NCH], a6[NCH];
    layer<K0, 1, 1, NCH, false>(0, a0, a1, st, W0, B0, M0, M0, Q_SHIFT);
    layer<KR, D1, NCH, NCH, true >(1, a1, a2, st, W1, B1, M1, MS1, Q_SHIFT);
    layer<KR, D2, NCH, NCH, true >(2, a2, a3, st, W2, B2, M2, MS2, Q_SHIFT);
    layer<KR, D3, NCH, NCH, true >(3, a3, a4, st, W3, B3, M3, MS3, Q_SHIFT);
    layer<KR, D4, NCH, NCH, true >(4, a4, a5, st, W4, B4, M4, MS4, Q_SHIFT);
    layer<1, 1, NCH, NCH, false>(5, a5, a6, st, W5, B5, M5, M5, Q_SHIFT);
    for (int c = 0; c < C_FEAT; ++c) {
#pragma HLS UNROLL
        st.S1[c] += a6[c];
        st.S2[c] += (int64_t)a6[c] * a6[c];
    }
    st.n += 1;
}
"""

TB_CPP = r"""#include <cstdio>
#include <vector>
#include "ccnn.h"

// input: n_traces T, then n_traces*T int values; output: per trace S1[C] S2[C]
int main(int argc, char **argv) {
    FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");
    int n, T; if (fscanf(fi, "%d %d", &n, &T) != 2) return 1;
    static CcnnState st;
    for (int i = 0; i < n; ++i) {
        ccnn_reset(st);
        for (int t = 0; t < T; ++t) { int v; if (fscanf(fi, "%d", &v) != 1) return 2; ccnn_step((int8_t)v, st); }
        for (int c = 0; c < C_FEAT; ++c) fprintf(fo, "%lld ", (long long)st.S1[c]);
        for (int c = 0; c < C_FEAT; ++c) fprintf(fo, "%lld ", (long long)st.S2[c]);
        fprintf(fo, "\n");
    }
    fclose(fi); fclose(fo); return 0;
}
"""


def _arr(name, a, ctype):
    flat = ", ".join(str(int(v)) for v in np.asarray(a).ravel())
    dims = "".join(f"[{d}]" for d in np.asarray(a).shape)
    return f"static const {ctype} {name}{dims} = {{{flat}}};\n"


def emit(q, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = q["layers"]
    shift = L[0]["shift"]
    assert all(l["shift"] == shift for l in L)
    C = L[1]["w"].shape[2]
    dil = [l["dil"] for l in L[1:5]]
    depth = max((l["w"].shape[0] - 1) * l["dil"] + 1 for l in L)
    h = ["#pragma once\n#include <cstdint>\n",
         f"#define N_LAYERS {len(L)}\n#define NCH {C}\n#define C_FEAT {C}\n#define MAX_C {C}\n",
         f"#define MAX_DEPTH {depth}\n#define K0 {L[0]['w'].shape[0]}\n#define KR {L[1]['w'].shape[0]}\n",
         "".join(f"#define D{i + 1} {d}\n" for i, d in enumerate(dil)),
         f"#define Q_SHIFT {shift}\n"]
    for i, l in enumerate(L):
        if np.any(np.abs(l["b"]) >= 2 ** 31):
            raise ValueError("bias exceeds int32")
        h.append(_arr(f"W{i}", l["w"], "int8_t"))
        h.append(_arr(f"B{i}", l["b"], "int32_t"))
        h.append(_arr(f"M{i}", l["m"], "int32_t"))
        if l["residual"]:
            h.append(_arr(f"MS{i}", l["m_skip"], "int32_t"))
    (out / "ccnn_weights.h").write_text("".join(h))
    (out / "ccnn.h").write_text(KERNEL_H)
    (out / "ccnn.cpp").write_text(KERNEL_CPP)
    (out / "tb.cpp").write_text(TB_CPP)
    return out
