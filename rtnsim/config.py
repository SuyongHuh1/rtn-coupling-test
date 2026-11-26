"""Configuration for the coupled-trap RTN simulator.

Physics
-------
K two-state traps, occupancy s_i in {0, 1}.  The free energy of a configuration
(in kT) is

    E(s, u) = sum_i eps_i s_i + sum_{i<j} J_ij s_i s_j + u * sum_i gamma_i s_i

where eps_i = ln(e_i / c_i) sets the uncoupled occupancy p_i = c_i / (c_i + e_i),
J is the kinetic (energetic) trap-trap interaction and u in {0, 1} is an optional
external telegraph drive (e.g. a gate-voltage or temperature fluctuation shared
by several traps).  With h_i = sum_j J_ij s_j + u gamma_i the rates of trap i are

    capture   c_i' = c_i exp(-phi h_i),     emission  e_i' = e_i exp((1 - phi) h_i)

so c_i'/e_i' = (c_i/e_i) exp(-h_i): detailed balance with respect to E at fixed u.
phi is the Bronsted splitting of the interaction between the two rates.

Readout: I(t) = I0 + sum_i A_i s_i + sum_{i<j} B_ij s_i s_j, sampled every dt
(point sample or box average over the last f*dt), plus white noise, optional
1/f-like flicker noise and a random-walk drift.  B is a readout (amplitude)
interaction, which changes the levels without changing the kinetics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json

import numpy as np


@dataclass
class TrapConfig:
    tau_c: List[float] = field(default_factory=lambda: [1e-3, 1e-3])  # s, mean empty dwell (uncoupled) = 1/c
    tau_e: List[float] = field(default_factory=lambda: [1e-3, 1e-3])  # s, mean filled dwell (uncoupled) = 1/e
    amp: List[float] = field(default_factory=lambda: [1.0, 0.6])      # step heights (a.u.)

    @property
    def K(self) -> int:
        return len(self.tau_c)


@dataclass
class CouplingConfig:
    J: Optional[List[List[float]]] = None   # K x K symmetric, kT; None = no kinetic coupling
    phi: float = 0.5                          # Bronsted splitting (fraction on capture)
    B: Optional[List[List[float]]] = None   # K x K readout interaction (upper triangle used)


@dataclass
class DriveConfig:
    """Shared telegraph drive u(t) in {0, 1} that shifts trap energies by gamma_i."""
    enabled: bool = False
    tau_off: float = 0.02   # s, mean dwell in u = 0
    tau_on: float = 0.02    # s, mean dwell in u = 1
    gamma: List[float] = field(default_factory=lambda: [1.0, 1.0])  # kT per trap


@dataclass
class ReadoutConfig:
    dt: float = 1e-4             # s, sampling interval
    integrate: float = 0.0       # 0 = point sampling; f in (0, 1] = box average over the last f*dt
    baseline: float = 0.0
    sigma: float = 0.2           # white noise sd per sample
    flicker_rms: float = 0.0     # rms of the 1/f-like component
    flicker_fmin: float = 1.0    # Hz
    flicker_fmax: float = 1e4    # Hz
    flicker_terms: int = 8       # Lorentzians, log-spaced corners, equal power per term
    drift_sd: float = 0.0        # random-walk drift, sd per sqrt(second)


@dataclass
class RTNConfig:
    traps: TrapConfig = field(default_factory=TrapConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    n_samples: int = 10000
    seed: int = 0

    @property
    def duration(self) -> float:
        return self.n_samples * self.readout.dt

    def J_matrix(self) -> np.ndarray:
        K = self.traps.K
        if self.coupling.J is None:
            return np.zeros((K, K))
        J = np.asarray(self.coupling.J, dtype=float)
        if J.shape != (K, K) or not np.allclose(J, J.T) or np.any(np.diag(J) != 0):
            raise ValueError("J must be a symmetric K x K matrix with zero diagonal")
        return J

    def B_matrix(self) -> np.ndarray:
        K = self.traps.K
        if self.coupling.B is None:
            return np.zeros((K, K))
        return np.triu(np.asarray(self.coupling.B, dtype=float), 1)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
