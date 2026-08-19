"""Immutable modeling configuration for Steps 2-5."""

from __future__ import annotations

SUPPORTED_STATE_COUNTS: tuple[int, int] = (2, 3)
HMM_SEEDS: tuple[int, int, int, int, int] = (42, 43, 44, 45, 46)
HMM_N_ITER: int = 500
HMM_TOL: float = 1e-6
HMM_MIN_COVAR: float = 1e-6
STATIONARY_TOL: float = 1e-10
PROBABILITY_TOL: float = 1e-8
LIKELIHOOD_TIE_TOL: float = 1e-12
BIC_TIE_TOL: float = 1e-12
HMM_MIN_STATE_OCCUPANCY: float = 0.05
