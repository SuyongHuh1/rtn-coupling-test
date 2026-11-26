"""Slide-size figures for every effect in the deck (R1-R18). Values from results/r8/*.json, logs and the
R8_report tables.  Output: results/figs/d_*.png.   python experiments/figs_deck.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "results/figs"; OUT.mkdir(exist_ok=True)
R8 = ROOT / "results/r8"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
OURS, GRAYD = S[0], "#8c8b86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": ["Noto Sans CJK KR", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200})
WFIG = (11.5, 4.0)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"d_{name}.png", dpi=200, facecolor="white"); plt.close(fig)


def bars(ax, groups, series, vals, colors, fmt="{:.2f}", ylim=None, log=False, label_vals=True):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (s, c) in enumerate(zip(series, colors)):
        v = np.array(vals[i], float)
        b = ax.bar(x - 0.4 + w * (i + 0.5), v, w - 0.03, color=c, label=s, edgecolor="white", linewidth=1.2)
        if label_vals:
            for xi, vi in zip(x - 0.4 + w * (i + 0.5), v):
                if np.isfinite(vi):
                    ax.text(xi, vi * (1.08 if log else 1) + (0 if log else (ylim[1] if ylim else max(map(np.nanmax, vals))) * 0.015),
                            fmt.format(vi), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
    ygrid(ax)


def five(ax, xmax):
    ax.axhline(0.05, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.text(xmax, 0.06, "5 %", fontsize=10, color=INK, ha="right", va="bottom")


# ---------------- R1: data needed vs J
def r1():
    J = np.array([0.3, 0.5, 1, 2]); N = np.array([1000, 424, 112, 35])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    jj = np.geomspace(0.25, 2.3, 50)
    ax.plot(jj, 112 / jj ** 2, color=GRAYD, lw=2, label="∝ 1/J²")
    ax.plot(J, N, "o", color=OURS, ms=9, mec="white", mew=1.5, label="이론 최소 (R1, 잡음 없음)")
    for a, b in zip(J, N):
        ax.text(a * 1.06, b * 1.08, f"{b}회", fontsize=11, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("결합 J (kT)"); ax.set_ylabel("트랩당 필요한 전환 수")
    ax.set_xticks([0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.3", "0.5", "1", "2"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ygrid(ax); ax.legend(loc="upper right")
    save(fig, "r1_transitions")


# ---------------- R2: efficiency
def r2():
    g = ["J=2, SNR 4", "J=2, SNR 2", "J=1, SNR 4", "J=1, SNR 2", "J=0.5, SNR 4"]
    v = [[0.50, 0.47, 0.52, 0.54, 0.54], [0.76, 0.21, 0.93, 0.47, 0.88], [0.48, 0.22, 0.30, 0.25, 0.25]]
    fig, ax = plt.subplots(figsize=WFIG)
    bars(ax, g, ["결합 HMM 검정", "상태 상관", "CNN"], v, S[:3], ylim=(0, 1.08))
    ax.set_ylabel("효율 (이론 최소 길이 ÷ 실제 필요 길이)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "r2_efficiency")


# ---------------- R3: confounds
def r3():
    g = ["느린 구동(같은 부호)", "1/f 잡음", "드리프트"]
    v = [[0.028, 0.013, 0.951], [0.011, 0.009, 0.902], [0.002, 0.004, 0.004], [0.001, 0.004, 0.005]]
    fig, ax = plt.subplots(figsize=WFIG)
    bars(ax, g, ["HMM 검정", "구동 인식 검정", "차분 CNN", "차분 CNN int8"], v, S[:4], fmt="{:.3f}", log=True, ylim=(5e-4, 3))
    five(ax, 2.45); ax.set_ylabel("오탐률 (로그)"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "r3_confounds")


# ---------------- R4: adaptive bias
def r4():
    g = ["J = 0.5 kT", "J = 1 kT"]
    med = [[1.50, 1.51], [2.99, 2.94], [1.23, 1.48]]; q90 = [[5.7, 5.1], [4.2, 4.0], [2.3, 2.6]]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(2); w = 0.8 / 3
    for i, (lab, c) in enumerate(zip(["고정 바이어스 V = 0", "9점 균등 스윕", "적응형"], [GRAYD, S[1], OURS])):
        xi = x - 0.4 + w * (i + 0.5)
        ax.bar(xi, med[i], w - 0.04, color=c, label=lab, edgecolor="white", linewidth=1.2)
        ax.errorbar(xi, med[i], yerr=[np.zeros(2), np.array(q90[i]) - np.array(med[i])], fmt="none", ecolor=INK2, capsize=4, lw=1)
        for a, b in zip(xi, med[i]): ax.text(a, b + 0.08, f"{b:.2f}", ha="center", fontsize=9)
    ax.axhline(1, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels(g); ax.set_ylabel("필요 기록 길이 (최적 고정 = 1)\n막대 = 중앙값, 선 = 90% 분위")
    ygrid(ax); ax.legend(loc="upper right")
    save(fig, "r4_adaptive")


# ---------------- R10b Zhang, R10c Zou
def zhang_zou():
    fig, (a, b) = plt.subplots(1, 2, figsize=WFIG)
    fpr = [0.33, 0.84, 0.10, 0.72]; tpr = [0.29, 0.89, 0.10, 0.70]; lab = ["흰색·2", "흰색·3", "1/f·2", "1/f·3"]
    a.plot([0, 1], [0, 1], color=GRAYD, lw=1.5, ls=(0, (4, 3)))
    a.scatter(fpr, tpr, s=90, color=S[6], edgecolor="white", lw=1.5, zorder=3)
    for f_, t_, l_ in zip(fpr, tpr, lab): a.text(f_ + 0.03, t_ - 0.06, l_, fontsize=10)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("오탐률 (결합 없는 기록)"); a.set_ylabel("검출률 (결합 있는 기록)")
    a.set_title("(a) Zhang 2017 규칙: 상태 수 < 2^K", fontsize=12, loc="left"); ygrid(a)
    a.text(0.55, 0.42, "대각선 = 판별력 없음", fontsize=10, color=INK2, rotation=0)
    g = ["트랩 1.4개\nSNR 4", "트랩 3.5개\nSNR 4", "트랩 1.4개\nSNR 10", "트랩 3.5개\nSNR 10"]
    v = [0.50, 0.75, 0.34, 0.58]
    b.bar(range(4), v, 0.6, color=S[1], edgecolor="white")
    for i, vi in enumerate(v): b.text(i, vi + 0.015, f"{vi:.0%}", ha="center", fontsize=10)
    b.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3))); b.text(3.4, 0.51, "Zou가 결합 근거로 본 50%", fontsize=9, ha="right")
    b.set_xticks(range(4)); b.set_xticklabels(g, fontsize=10); b.set_ylim(0, 0.9); b.set_ylabel("상태 소실률 (결합 없는 집단)")
    b.set_title("(b) Zou 2014 재현: 독립 트랩에서도 상태가 사라짐", fontsize=12, loc="left"); ygrid(b)
    save(fig, "zhang_zou")


# ---------------- robustness
def robust():
    g = ["비지수 머무름\n(30개)", "진폭 결합\n(흰색)", "진폭 결합\n(1/f)", "공통 구동\n(조건 A, 15개)", "공통 구동\n(조건 B, 15개)"]
    v = [0 / 30, 0.05, 0.025, 0 / 15, 1 / 15]
    fig, ax = plt.subplots(figsize=WFIG)
    ax.bar(range(5), v, 0.6, color=OURS, edgecolor="white")
    for i, vi in enumerate(v): ax.text(i, vi + 0.004, f"{vi:.3f}", ha="center", fontsize=10)
    five(ax, 4.4); ax.set_xticks(range(5)); ax.set_xticklabels(g); ax.set_ylim(0, 0.15); ax.set_ylabel("우리 방법 오탐률"); ygrid(ax)
    save(fig, "robust")


# ---------------- R11 Gate_P, R11b LH
def nichol():
    fig, (a, b) = plt.subplots(1, 2, figsize=WFIG, gridspec_kw=dict(width_ratios=[1.25, 1]))
    V = ["0.5630", "0.5635", "0.5640", "0.5645", "0.5650", "0.5655", "0.5660", "0.5665", "0.5670"]
    L = [1.61, 0.01, 10.44, 1.21, 7.68, 14247, 2848, 38.6, 0.26]
    col = [GRAYD] * 5 + [S[7], S[7], S[3], OURS]
    a.bar(range(9), L, 0.65, color=col, edgecolor="white"); a.set_yscale("log"); a.set_ylim(5e-3, 1e5)
    for i, l_ in enumerate(L): a.text(i, l_ * 1.3, f"{l_:g}", ha="center", fontsize=8.5)
    a.set_xticks(range(9)); a.set_xticklabels([v[2:] for v in V], rotation=45, fontsize=10)
    a.set_xlabel("RP1 (V, 0.5 생략)"); a.set_ylabel("결합 LRT (로그)"); ygrid(a)
    a.set_title("(a) Gate_P: 회색 근거 없음 · 빨강 판정 불가(박스카)\n노랑 귀속 안 함 · 파랑 0.567 V(Albrecht 주장) 근거 없음", fontsize=10.5, loc="left")
    S_ = ["0.1480", "0.1490", "0.1505", "0.1510", "0.1520"]
    drive = [1766, 5472, 1521, 1112, 465]; after = [97, 396, 211, 569, 58]
    x = np.arange(5)
    b.bar(x - 0.2, drive, 0.38, color=S[1], label="숨은 상태 LRT", edgecolor="white")
    b.bar(x + 0.2, after, 0.38, color=OURS, label="숨은 상태 포함 후 결합 LRT", edgecolor="white")
    b.set_yscale("log"); b.set_xticks(x); b.set_xticklabels(S_, fontsize=10); b.set_xlabel("S (V)"); ygrid(b)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=10)
    save(fig, "nichol")


# ---------------- R12
def r12():
    g = ["Guo식 Δτ/τ\n|d|>0.5", "Albrecht\n(원문 절차)", "Son\n(원문 절차)", "우리\n점근 χ²", "우리\n파이프라인 보정"]
    alb = json.load(open(R8 / "r19b_albrecht_realneg.json"))["summary"]["alb_full"]
    son = json.load(open(R8 / "r19c_son_realneg.json"))["summary"]["son_faithful_fpr"]
    v = [0.95, alb, son, 0.075, 0.0]
    fig, ax = plt.subplots(figsize=WFIG)
    ax.bar(range(5), v, 0.6, color=[S[1], S[2], S[3], S[4], OURS], edgecolor="white")
    for i, vi in enumerate(v): ax.text(i, vi + 0.015, ("0/40" if vi == 0 else (f"{vi:.3f}" if vi < 0.1 else f"{vi:.2f}")), ha="center", fontsize=11)
    five(ax, 4.45); ax.set_xticks(range(5)); ax.set_xticklabels(g); ax.set_ylim(0, 1.12); ax.set_ylabel("오탐률 (실측 잡음 기록 40개)"); ygrid(ax)
    save(fig, "r12_realneg")


# ---------------- R15
def r15():
    W_ = json.load(open(R8 / "r15_rtninja_white.json"))["summary"]; F_ = json.load(open(R8 / "r15_rtninja_full.json"))["summary"]
    R10w = json.load(open(R8 / "r10_baselines_white.json"))["summary"]; R10f = json.load(open(R8 / "r10_baselines_full.json"))["summary"]
    g = ["흰색·트랩 2", "흰색·트랩 3", "1/f·트랩 2", "1/f·트랩 3"]
    Sw = json.load(open(R8 / "r19c_son_white.json"))["summary"]; Sf = json.load(open(R8 / "r19c_son_full.json"))["summary"]
    son_f = [Sw["K=2"]["fpr"], Sw["K=3"]["fpr"], Sf["K=2"]["fpr"], Sf["K=3"]["fpr"]]
    son_r = [W_["K=2"]["son_faithful"]["fpr"], W_["K=3"]["son_faithful"]["fpr"], F_["K=2"]["son_faithful"]["fpr"], F_["K=3"]["son_faithful"]["fpr"]]
    guo_r = [W_["K=2"]["guo_d0p5"]["fpr"], W_["K=3"]["guo_d0p5"]["fpr"], F_["K=2"]["guo_d0p5"]["fpr"], F_["K=3"]["guo_d0p5"]["fpr"]]
    fig, ax = plt.subplots(figsize=WFIG)
    bars(ax, g, ["Son(원문 절차), FHMM 분해", "Son(원문 절차), RTNinja 재구현 분해", "Guo식, RTNinja 재구현 분해"], [son_f, son_r, guo_r],
         [S[3], S[1], S[6]], ylim=(0, 1.15))
    five(ax, 3.45); ax.set_ylabel("오탐률"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=10.5)
    save(fig, "r15")


# ---------------- R16
def r16():
    fig, ax = plt.subplots(figsize=WFIG)
    for k, (cond, c) in enumerate([("white", S[2]), ("full", S[6])]):
        d = json.load(open(R8 / f"r16_guo_amp_{cond}.json"))["res"]
        s = np.array([v for r in d.values() for v in r["s"]], float); s = s[np.isfinite(s)]
        ax.hist(np.clip(s, -1.5, 1.5), bins=np.linspace(-1.5, 1.5, 61), histtype="step", lw=2, color=c,
                label=("흰색 잡음" if cond == "white" else "1/f + 드리프트") + f" (트랩 쌍 {s.size}개)", density=True)
    ax.axvspan(-0.25, -0.1, color=S[1], alpha=0.18, lw=0)
    ax.text(-0.27, ax.get_ylim()[1] * 0.92, "Guo 2018이 보고한\n집단 평균 범위", ha="right", fontsize=10, color=INK)
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("진폭 결합 세기 s = λc/λ − 1 (참값 0, ±1.5에서 자름)"); ax.set_ylabel("밀도"); ygrid(ax); ax.legend(loc="upper right")
    save(fig, "r16_guo_amp")


# ---------------- R17
def r17():
    A = json.load(open(R8 / "r17_partA.json"))
    fig, ax = plt.subplots(figsize=WFIG)
    g = ["2소스 (판정 대상 68 / RTNinja 53)", "3소스 (판정 대상 22 / RTNinja 12)"]
    # rules restricted to checklist-testable records (computed in R17), ours with checklist
    guo = [0.79, 1.00]; son = [0.32, 0.58]; zh = [0.28, 0.67]; ours_a = [4 / 68, 4 / 22]; ours_c = [0 / 6, 2 / 22]
    bars(ax, g, ["RTNinja + Guo식 Δτ/τ", "RTNinja + Son(원문 절차)", "RTNinja + Zhang 레벨 수", "우리 (체크리스트 + 점근 χ²)", "우리 (체크리스트 + 보정)"],
         [guo, son, zh, ours_a, ours_c], [S[1], S[3], S[6], S[4], OURS], ylim=(0, 1.15))
    five(ax, 1.45); ax.set_ylabel("오탐률 (모두 결합 없는 기록)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=10)
    save(fig, "r17_fpr")
    # all-record rule FPR (no checklist)
    fig, ax = plt.subplots(figsize=WFIG)
    k2, k3 = A["2source"]["K_correct"], A["3source"]["K_correct"]
    D = json.load(open(R8 / "r19d_son_rtninja.json"))
    bars(ax, ["2소스 (RTNinja K 정답 120)", "3소스 (RTNinja K 정답 82)"], ["Guo식 Δτ/τ", "Son(원문 절차)", "Zhang 레벨 수"],
         [[k2["guo_d0p5"], k3["guo_d0p5"]], [D["2source"]["K_correct"]["fpr"], D["3source"]["K_correct"]["fpr"]], [k2["zhang_lv"], k3["zhang_lv"]]], [S[1], S[3], S[6]], ylim=(0, 1.15))
    five(ax, 1.45); ax.set_ylabel("오탐률"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "r17_rules_all")
    # funnel
    fig, ax = plt.subplots(figsize=WFIG)
    cats = ["적합 실패\n(수치 오류)", "검정 대상 아님\n(전환 15회 미만)", "판정 불가\n(머무름 < 3샘플)", "근거 없음", "결합 판정 (오탐)"]
    c2 = [18, 63, 51, 64, 4]; c3 = [34, 51, 93, 18, 4]
    cols = [GRAYD, S[3], S[7], OURS, S[1]]
    for row, (lab, cs) in enumerate([("2소스", c2), ("3소스", c3)]):
        left = 0
        for c_, n_, cat in zip(cols, cs, cats):
            ax.barh(row, n_, left=left, color=c_, edgecolor="white", lw=2, label=cat if row == 0 else None)
            if n_ >= 8: ax.text(left + n_ / 2, row, str(n_), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
            else: ax.text(left + n_ + 1.5, row, str(n_), ha="left", va="center", color=INK, fontsize=11, fontweight="bold")
            left += n_
    ax.set_yticks([0, 1]); ax.set_yticklabels(["2소스 (200)", "3소스 (200)"]); ax.invert_yaxis(); ax.set_xlabel("기록 수")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=10)
    save(fig, "r17_funnel")
    # agreement
    C = json.load(open(R8 / "r17c_reimpl_vs_rtninja.json"))
    fig, ax = plt.subplots(figsize=WFIG)
    for k, c in (("2source", OURS), ("3source", S[1])):
        ag = np.array([r["agree"] for r in C[k]["rows"]])
        ax.hist(ag, bins=np.linspace(0.4, 1.0, 31), histtype="step", lw=2, color=c,
                label=f"{k[0]}소스 ({ag.size}개, 중앙값 {np.median(ag):.3f})")
    ax.set_xlabel("R15 재구현과 RTNinja 원본의 샘플별 소스 활동 일치율"); ax.set_ylabel("기록 수"); ygrid(ax); ax.legend(loc="upper left")
    save(fig, "r17_agree")


# ---------------- R19: original-text re-implementation vs earlier reproduction
def r19():
    W = json.load(open(R8 / "r10_baselines_white.json"))["summary"]; F = json.load(open(R8 / "r10_baselines_full.json"))["summary"]
    AW = json.load(open(R8 / "r19_albrecht_white.json"))["summary"]; AF = json.load(open(R8 / "r19_albrecht_full.json"))["summary"]
    SW = json.load(open(R8 / "r19c_son_white.json"))["summary"]; SF = json.load(open(R8 / "r19c_son_full.json"))["summary"]
    RN = json.load(open(R8 / "r12_realneg.json"))["summary"]
    ab = json.load(open(R8 / "r19b_albrecht_realneg.json"))["summary"]["alb_full"]
    sb = json.load(open(R8 / "r19c_son_realneg.json"))["summary"]["son_faithful_fpr"]
    g = ["흰색·트랩 2", "흰색·트랩 3", "1/f·트랩 2", "1/f·트랩 3", "실측 잡음 40개"]
    ks = [(W, "K=2"), (W, "K=3"), (F, "K=2"), (F, "K=3")]
    a_old = [d[k]["albrecht_e2"]["fpr"] for d, k in ks] + [RN["albrecht_e2"]]
    a_new = [AW["K=2"]["alb_full"]["fpr"], AW["K=3"]["alb_full"]["fpr"], AF["K=2"]["alb_full"]["fpr"], AF["K=3"]["alb_full"]["fpr"], ab]
    s_old = [d[k]["son_bonf"]["fpr"] for d, k in ks] + [RN["son_bonf"]]
    s_new = [SW["K=2"]["fpr"], SW["K=3"]["fpr"], SF["K=2"]["fpr"], SF["K=3"]["fpr"], sb]
    fig, (a, b) = plt.subplots(1, 2, figsize=WFIG, sharey=True)
    bars(a, g, ["기존 재현 (준위 자유, e > 2)", "원문 절차 (준위 고정 + 부트스트랩)"], [a_old, a_new], [GRAYD, S[2]], ylim=(0, 1.15))
    a.axhline(0.05, color=INK, lw=0.9, ls=(0, (4, 3))); a.text(0.5, 0.06, "5 %", fontsize=10, color=INK, ha="center", va="bottom"); a.set_title("(a) Albrecht 2026", loc="left", fontsize=12); a.set_ylabel("오탐률 (결합 없는 기록)")
    a.legend(loc="upper left", fontsize=9.5); a.tick_params(axis="x", labelsize=9)
    bars(b, g, ["기존 재현 (모든 쌍 χ² + Bonferroni)", "원문 절차 (트랩2 | 트랩1 한 번)"], [s_old, s_new], [GRAYD, S[3]], ylim=(0, 1.15))
    b.axhline(0.05, color=INK, lw=0.9, ls=(0, (4, 3))); b.text(0.5, 0.06, "5 %", fontsize=10, color=INK, ha="center", va="bottom"); b.set_title("(b) Son 2017", loc="left", fontsize=12); b.legend(loc="upper left", fontsize=9.5); b.tick_params(axis="x", labelsize=9)
    save(fig, "r19_faithful")


# ---------------- detection power at matched FPR (R10 records)
def power():
    W = json.load(open(R8 / "r10_baselines_white.json"))["summary"]; F = json.load(open(R8 / "r10_baselines_full.json"))["summary"]
    P = json.load(open(R8 / "r19_power_faithful.json"))
    ks = [("white", "K=2"), ("white", "K=3"), ("full", "K=2"), ("full", "K=3")]
    D = {"white": W, "full": F}
    g = ["흰색·트랩 2", "흰색·트랩 3", "1/f·트랩 2", "1/f·트랩 3"]
    guo = [D[c][k]["tpr5_pku"] for c, k in ks]
    alb = [P[f"{c} {k}"]["alb"]["tpr5"] for c, k in ks]
    son = [P[f"{c} {k}"]["son"]["tpr5"] for c, k in ks]
    ours = [D[c][k]["tpr5_ours"] for c, k in ks]
    upper = [0.88, 0.83, 0.88, 0.83]
    fig, (a, b) = plt.subplots(1, 2, figsize=WFIG, gridspec_kw=dict(width_ratios=[1.6, 1]))
    bars(a, g, ["Guo식", "Albrecht (원문)", "Son (원문)", "우리 CNN–HMM", "상한 (참 상태를 알 때)"], [guo, alb, son, ours, upper],
         [S[1], S[2], S[3], OURS, "#c9c8c3"], ylim=(0, 1.08), label_vals=False)
    x = np.arange(4); w = 0.8 / 5
    for xi, v in zip(x - 0.4 + w * 3.5, ours): a.text(xi - 0.03, v + 0.02, f"{v:.2f}", ha="right", fontsize=9.5, color=INK, fontweight="bold")
    a.set_ylabel("검출률 (오탐 5%에 맞춘 문턱)"); a.set_title("(a) 같은 오탐 5%에서 진짜 결합을 잡는 비율", loc="left", fontsize=12)
    a.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=9.5); a.tick_params(axis="x", labelsize=10)
    aucg = [D[c][k]["auc_pku"] for c, k in ks], [P[f"{c} {k}"]["alb"]["auc"] for c, k in ks], [P[f"{c} {k}"]["son"]["auc"] for c, k in ks], [D[c][k]["auc_ours"] for c, k in ks]
    bars(b, g, ["Guo식", "Albrecht", "Son", "우리"], list(aucg), [S[1], S[2], S[3], OURS], ylim=(0.4, 1.02), label_vals=False)
    for xi, v in zip(x - 0.4 + 0.2 * 3.5, aucg[3]): b.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    b.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3))); b.text(3.45, 0.485, "0.5 = 무작위", fontsize=9, ha="right", va="top")
    b.set_ylabel("AUC (판별력)"); b.set_title("(b) 문턱과 무관한 판별력", loc="left", fontsize=12); b.tick_params(axis="x", labelsize=9)
    save(fig, "power")


# ---------------- R18
def r18():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    J = np.linspace(-1.5, 3, 200); p11r, var = [], []
    for j in J:
        w = np.array([1, 1, 1, np.exp(-j)]); p = w / w.sum(); p1 = p[2] + p[3]; cov = p[3] - p1 * p1
        p11r.append(p[3] / p1 ** 2); var.append((2 * p1 * (1 - p1) + 2 * cov) / (2 * p1 * (1 - p1)))
    ax.plot(J, p11r, color=OURS, lw=2.5, label="둘 다 찬 확률 (독립 대비)")
    ax.plot(J, var, color=S[1], lw=2.5, label="전류 요동 분산 (독립 대비)")
    ax.axhline(1, color=INK, lw=0.9, ls=(0, (4, 3))); ax.axvspan(-0.1, 0.1, color=GRAYD, alpha=0.2, lw=0)
    ax.text(0.13, 1.45, "|J| ≤ 0.1: 0.567 V에서\n배제한 범위", fontsize=9.5)
    ax.set_xlabel("J (kT)  ← 협력 | 반발 →"); ax.set_ylabel("독립 모델 대비 비율"); ygrid(ax); ax.legend(loc="lower left", fontsize=10.5)
    save(fig, "r18_pair")
    R = json.load(open(R8 / "r18c_smooth.json"))
    cmap = LinearSegmentedColormap.from_list("div", ["#c6402f", "#f0efec", "#2a78d6"])
    fig, axs = plt.subplots(1, 2, figsize=WFIG)
    for ax, Wv, ttl in zip(axs, (1.0, 4.0), ("트랩 대부분 활동 (W = 1 kT)", "일부만 활동 (W = 4 kT)")):
        fs = (0.1, 0.3, 1.0); Js = (0.25, 0.5, 1.0, 2.0)
        M = np.array([[[x for x in R if x["amp"] == "exp" and x["W"] == Wv and x["sign"] == "repulsive" and x["f"] == f and x["Jm"] == j][0]["tail_ratio_1e-06"] for j in Js] for f in fs])
        im = ax.imshow(np.log2(np.clip(M, 1 / 64, 64)), cmap=cmap, norm=TwoSlopeNorm(0, -4, 4), aspect="auto")
        for i in range(3):
            for k in range(4):
                v = M[i, k]; first = v < 0.8 or v > 1.25
                ax.text(k, i, f"{v:.2f}" + (" ●" if first else ""), ha="center", va="center", fontsize=11, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{j:g}" for j in Js]); ax.set_yticks(range(3)); ax.set_yticklabels([f"{f:.0%}" for f in fs])
        ax.set_xlabel("평균 결합 세기 Jm (kT)"); ax.set_ylabel("결합 쌍 비율 f"); ax.set_title(ttl, fontsize=12, loc="left")
    fig.text(0.5, 0.005, "값 = 고장 확률(10⁻⁶ 꼬리) 독립 대비 비율, 반발 결합, 계단 지수분포.  ● = 1차 효과(25% 이상 변화)", ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "d_r18_heat.png", dpi=200, facecolor="white"); plt.close(fig)
    # margin shift
    fig, ax = plt.subplots(figsize=WFIG)
    for c, (f, Wv) in zip([S[0], S[1], S[2], S[3]], [(0.3, 1.0), (1.0, 1.0), (0.3, 4.0), (1.0, 4.0)]):
        Js = (0.25, 0.5, 1.0, 2.0)
        tr = [[x for x in R if x["amp"] == "exp" and x["W"] == Wv and x["sign"] == "repulsive" and x["f"] == f and x["Jm"] == j][0] for j in Js]
        ax.plot(Js, [abs(t["q_shift_1e-06"]) * 100 for t in tr], "o-", color=c, lw=2, ms=7, mec="white", label=f"f = {f:.0%}, W = {Wv:g}")
    ax.axhline(5, color=INK, lw=0.9, ls=(0, (4, 3))); ax.text(2.0, 5.3, "5% (1차 기준)", ha="right", fontsize=10)
    ax.set_xlabel("평균 결합 세기 Jm (kT)"); ax.set_ylabel("필요 전압 마진 변화 (%)\n(10⁻⁶ 분위수)"); ygrid(ax); ax.legend(ncol=2)
    save(fig, "r18_margin")


if __name__ == "__main__":
    for f in ([globals()[a] for a in sys.argv[1:]] or (r1, r2, r3, r4, zhang_zou, robust, nichol, r12, r15, r16, r17, r18, r19, power)):
        f(); print("ok", f.__name__)
