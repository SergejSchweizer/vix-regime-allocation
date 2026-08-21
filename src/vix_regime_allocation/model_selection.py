"""HMM-only selection with temporary legacy mixed-family compatibility wrappers."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from .hmm_model import HMMFitResult
from .model_config import BIC_TIE_TOL, HMM_MIN_STATE_OCCUPANCY, PROBABILITY_TOL

COMPARISON_COLUMNS: tuple[str, ...] = (
    "family",
    "n_states",
    "log_likelihood",
    "n_parameters",
    "n_observations",
    "aic",
    "bic",
    "converged",
    "min_viterbi_occupancy",
    "valid",
)
LEGACY_COMPARISON_COLUMNS: tuple[str, ...] = (
    "family",
    "n_states",
    "log_likelihood",
    "n_parameters",
    "n_observations",
    "aic",
    "bic",
    "converged",
    "criterion_scope",
)
_SUPPORTED_STATES: tuple[int, int] = (2, 3)


def _validated_hmm_candidates(
    candidates: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    if not isinstance(candidates, list):
        raise TypeError("hmm_candidates must be a list.")
    if len(candidates) != len(_SUPPORTED_STATES):
        raise ValueError("HMM candidates must contain exactly K=2 and K=3.")

    by_states: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Each HMM candidate must be a dictionary.")
        if candidate.get("family") != "hmm":
            raise ValueError("Candidate family must be exactly 'hmm'.")
        n_states = candidate.get("n_states")
        if isinstance(n_states, bool) or not isinstance(n_states, int):
            raise ValueError("Candidate n_states must be an integer.")
        if n_states not in _SUPPORTED_STATES or n_states in by_states:
            raise ValueError("HMM candidates must contain unique K=2 and K=3 entries.")
        for key in ("log_likelihood", "aic", "bic"):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Candidate {key} must be numeric.")
        for key in ("n_parameters", "n_observations"):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Candidate {key} must be a positive integer.")
        if not isinstance(candidate.get("converged"), bool):
            raise ValueError("Candidate converged must be Boolean.")
        by_states[n_states] = candidate
    return by_states


def _minimum_occupancy(candidate: dict[str, object]) -> float:
    fit = candidate.get("fit")
    n_states = candidate.get("n_states")
    if not isinstance(fit, HMMFitResult) or not isinstance(n_states, int):
        return float("nan")
    try:
        states = fit.states.to_numpy(dtype=int)
    except (TypeError, ValueError):
        return float("nan")
    if len(states) == 0 or np.any((states < 0) | (states >= n_states)):
        return float("nan")
    counts = np.bincount(states, minlength=n_states).astype(float)
    return float(counts.min() / len(states))


def _hmm_invalid_reason(candidate: dict[str, object]) -> str | None:
    if candidate.get("converged") is not True:
        return "candidate did not converge"
    log_likelihood = candidate.get("log_likelihood")
    if isinstance(log_likelihood, bool) or not isinstance(log_likelihood, (int, float)):
        return "candidate likelihood is invalid"
    if not math.isfinite(float(log_likelihood)):
        return "candidate likelihood is not finite"

    fit = candidate.get("fit")
    if not isinstance(fit, HMMFitResult):
        return "candidate fit result is missing or invalid"
    if not fit.converged:
        return "selected HMM fit did not converge"

    n_states = int(cast(int, candidate["n_states"]))
    if fit.n_states != n_states:
        return "fit state count does not match candidate"

    means = fit.means.to_numpy(dtype=float)
    if len(means) != n_states or np.any(~np.isfinite(means)):
        return "state means are not finite"
    if np.any(np.diff(means) < 0.0):
        return "state means are not in canonical increasing order"

    variances = fit.variances.to_numpy(dtype=float)
    if len(variances) != n_states or np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        return "state variances are not finite and strictly positive"

    start = np.asarray(fit.start_probabilities, dtype=float)
    if (
        len(start) != n_states
        or np.any(~np.isfinite(start))
        or np.any(start < -PROBABILITY_TOL)
        or np.any(start > 1.0 + PROBABILITY_TOL)
        or not np.isclose(start.sum(), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        return "initial-state probabilities are invalid"

    transition = fit.transition_matrix.to_numpy(dtype=float)
    if (
        transition.shape != (n_states, n_states)
        or np.any(~np.isfinite(transition))
        or np.any(transition < -PROBABILITY_TOL)
        or np.any(transition > 1.0 + PROBABILITY_TOL)
        or not np.allclose(transition.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        return "transition probabilities are invalid"

    try:
        states = fit.states.to_numpy(dtype=int)
    except (TypeError, ValueError):
        return "decoded state path is invalid"
    if len(states) == 0 or np.any((states < 0) | (states >= n_states)):
        return "decoded state path is invalid"
    if set(np.unique(states)) != set(range(n_states)):
        return "decoded state path does not contain every configured state"

    posterior = fit.probabilities.to_numpy(dtype=float)
    if (
        posterior.shape != (len(states), n_states)
        or np.any(~np.isfinite(posterior))
        or np.any(posterior < -PROBABILITY_TOL)
        or np.any(posterior > 1.0 + PROBABILITY_TOL)
        or not np.allclose(posterior.sum(axis=1), 1.0, atol=PROBABILITY_TOL, rtol=0.0)
    ):
        return "posterior probabilities are invalid"
    if not fit.probabilities.index.equals(fit.states.index):
        return "posterior and decoded-state dates are misaligned"

    if _minimum_occupancy(candidate) < HMM_MIN_STATE_OCCUPANCY:
        return "at least one decoded HMM state has occupancy below 5%"
    return None


def build_hmm_model_comparison(hmm_candidates: list[dict[str, object]]) -> pd.DataFrame:
    """Build the exact two-row HMM K=2/K=3 comparison table."""
    candidates = _validated_hmm_candidates(hmm_candidates)
    rows: list[dict[str, object]] = []
    for n_states in _SUPPORTED_STATES:
        candidate = candidates[n_states]
        rows.append(
            {
                "family": "hmm",
                "n_states": n_states,
                "log_likelihood": float(cast(float, candidate["log_likelihood"])),
                "n_parameters": int(cast(int, candidate["n_parameters"])),
                "n_observations": int(cast(int, candidate["n_observations"])),
                "aic": float(cast(float, candidate["aic"])),
                "bic": float(cast(float, candidate["bic"])),
                "converged": bool(candidate["converged"]),
                "min_viterbi_occupancy": _minimum_occupancy(candidate),
                "valid": _hmm_invalid_reason(candidate) is None,
            }
        )
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def _validate_hmm_comparison(comparison: pd.DataFrame) -> None:
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) != COMPARISON_COLUMNS:
        raise ValueError("comparison columns do not match the canonical HMM-only schema.")
    if len(comparison) != 2:
        raise ValueError("comparison must contain exactly two HMM candidate rows.")
    if comparison["family"].astype(str).tolist() != ["hmm", "hmm"]:
        raise ValueError("comparison family must be HMM only.")
    if comparison["n_states"].astype(int).tolist() != [2, 3]:
        raise ValueError("comparison rows must be ordered HMM K=2 then HMM K=3.")
    valid_values: list[object] = comparison["valid"].tolist()
    if not all(isinstance(value, bool) for value in valid_values):
        raise ValueError("comparison valid column must contain Boolean values.")


def select_preferred_hmm(
    comparison: pd.DataFrame, hmm_candidates: list[dict[str, object]]
) -> dict[str, object]:
    """Select the valid HMM candidate with minimum BIC; never fall back to another family."""
    _validate_hmm_comparison(comparison)
    candidates = _validated_hmm_candidates(hmm_candidates)
    valid_rows = comparison.loc[comparison["valid"].astype(bool), ["n_states", "bic"]].copy()
    if valid_rows.empty:
        reasons = [
            f"K={k}: {_hmm_invalid_reason(candidates[k]) or 'invalid diagnostics'}"
            for k in _SUPPORTED_STATES
        ]
        raise RuntimeError("No valid HMM candidate is available; " + "; ".join(reasons))
    valid_rows = valid_rows.loc[np.isfinite(valid_rows["bic"].to_numpy(dtype=float))]
    if valid_rows.empty:
        raise RuntimeError("No valid HMM candidate has a finite BIC.")

    best_bic = float(valid_rows["bic"].min())
    tied = valid_rows.loc[
        valid_rows["bic"].astype(float).sub(best_bic).abs() <= BIC_TIE_TOL, "n_states"
    ]
    selected_n_states = int(tied.min())
    selected_candidate = candidates[selected_n_states]
    reason = _hmm_invalid_reason(selected_candidate)
    if reason is not None:
        raise RuntimeError(f"Comparison marked HMM K={selected_n_states} valid but {reason}.")
    fit = selected_candidate.get("fit")
    if not isinstance(fit, HMMFitResult):  # pragma: no cover - guarded above
        raise RuntimeError("Validated HMM candidate unexpectedly lacks its fit result.")
    states = fit.states.copy()
    states.name = "state"
    return {
        "family": "hmm",
        "n_states": selected_n_states,
        "states": states,
        "state_source": f"reports/tables/step2_hmm_{selected_n_states}_states.csv",
        "selection_reason": (
            f"Among valid HMM candidates, BIC selected K={selected_n_states}; "
            "a BIC tie within tolerance is resolved toward the lower state count."
        ),
    }


# The following mixed-family wrappers preserve the pre-revision canonical artifacts until
# PR-57/PR-60 replace the old analysis pipeline. They are not used by the HMM-only API above.
def _validated_legacy_candidates(
    candidates: list[dict[str, object]], family: str
) -> dict[int, dict[str, object]]:
    if len(candidates) != len(_SUPPORTED_STATES):
        raise ValueError(f"{family} candidates must contain exactly K=2 and K=3.")
    by_states: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Each candidate must be a dictionary.")
        if candidate.get("family") != family:
            raise ValueError(f"Candidate family must be {family!r}.")
        n_states = candidate.get("n_states")
        if isinstance(n_states, bool) or not isinstance(n_states, int):
            raise ValueError("Candidate n_states must be an integer.")
        if n_states not in _SUPPORTED_STATES or n_states in by_states:
            raise ValueError(f"{family} candidates must contain unique K=2 and K=3 entries.")
        for key in ("log_likelihood", "aic", "bic"):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Candidate {key} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Candidate {key} must be finite.")
        for key in ("n_parameters", "n_observations"):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Candidate {key} must be a positive integer.")
        if not isinstance(candidate.get("converged"), bool):
            raise ValueError("Candidate converged must be Boolean.")
        by_states[n_states] = candidate
    return by_states


def build_model_comparison(
    markov_candidates: list[dict[str, object]], hmm_candidates: list[dict[str, object]]
) -> pd.DataFrame:
    """Legacy four-row comparison retained solely for old artifact parity during migration."""
    markov = _validated_legacy_candidates(markov_candidates, "markov")
    hmm = _validated_legacy_candidates(hmm_candidates, "hmm")
    rows: list[dict[str, object]] = []
    for family, candidates in (("markov", markov), ("hmm", hmm)):
        for n_states in _SUPPORTED_STATES:
            candidate = candidates[n_states]
            rows.append(
                {
                    "family": family,
                    "n_states": n_states,
                    "log_likelihood": float(cast(float, candidate["log_likelihood"])),
                    "n_parameters": int(cast(int, candidate["n_parameters"])),
                    "n_observations": int(cast(int, candidate["n_observations"])),
                    "aic": float(cast(float, candidate["aic"])),
                    "bic": float(cast(float, candidate["bic"])),
                    "converged": bool(candidate["converged"]),
                    "criterion_scope": "within_family_only",
                }
            )
    return pd.DataFrame(rows, columns=list(LEGACY_COMPARISON_COLUMNS))


def _validate_legacy_comparison(comparison: pd.DataFrame) -> None:
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) != LEGACY_COMPARISON_COLUMNS:
        raise ValueError("legacy comparison columns do not match the pre-revision schema.")
    if len(comparison) != 4:
        raise ValueError("legacy comparison must contain exactly four candidate rows.")
    expected_pairs = {("markov", 2), ("markov", 3), ("hmm", 2), ("hmm", 3)}
    actual_pairs = set(
        zip(comparison["family"].astype(str), comparison["n_states"].astype(int), strict=True)
    )
    if actual_pairs != expected_pairs:
        raise ValueError("legacy comparison must contain Markov/HMM K=2/K=3 exactly once.")
    if not (comparison["criterion_scope"] == "within_family_only").all():
        raise ValueError("legacy criteria may only be interpreted within model family.")


def _legacy_best_within_family(comparison: pd.DataFrame, family: str) -> int:
    rows = comparison.loc[comparison["family"] == family, ["n_states", "bic"]].sort_values(
        "n_states", kind="stable"
    )
    best_bic = float(rows["bic"].min())
    tied = rows.loc[rows["bic"].astype(float).sub(best_bic).abs() <= BIC_TIE_TOL, "n_states"]
    return int(tied.min())


def select_preferred_model(
    comparison: pd.DataFrame,
    markov_candidates: list[dict[str, object]],
    hmm_candidates: list[dict[str, object]],
) -> dict[str, object]:
    """Legacy valid-HMM-or-Markov rule retained only until the canonical rebuild PRs land."""
    _validate_legacy_comparison(comparison)
    markov = _validated_legacy_candidates(markov_candidates, "markov")
    hmm = _validated_legacy_candidates(hmm_candidates, "hmm")
    markov_best = _legacy_best_within_family(comparison, "markov")
    hmm_best = _legacy_best_within_family(comparison, "hmm")
    selected_hmm = hmm[hmm_best]
    invalid_reason = _hmm_invalid_reason(selected_hmm)
    if invalid_reason is None:
        selected_family = "hmm"
        selected_n_states = hmm_best
        selected_candidate = selected_hmm
        reason = (
            f"Within-family BIC selected HMM K={hmm_best}; all fixed HMM validity "
            "diagnostics passed."
        )
    else:
        selected_family = "markov"
        selected_n_states = markov_best
        selected_candidate = markov[markov_best]
        reason = (
            f"Within-family BIC selected HMM K={hmm_best}, but {invalid_reason}; "
            f"the fixed fallback selects Markov K={markov_best}."
        )
    if selected_family == "hmm":
        fit = selected_candidate.get("fit")
        if not isinstance(fit, HMMFitResult):  # pragma: no cover - guarded above
            raise RuntimeError("Validated HMM candidate unexpectedly lacks its fit result.")
        states = fit.states.copy()
    else:
        candidate_states = selected_candidate.get("states")
        if not isinstance(candidate_states, pd.Series):
            raise ValueError("Selected Markov candidate must contain its canonical state Series.")
        states = candidate_states.copy()
    states.name = "state"
    return {
        "family": selected_family,
        "n_states": selected_n_states,
        "states": states,
        "state_source": f"reports/tables/step2_{selected_family}_{selected_n_states}_states.csv",
        "selection_reason": reason,
        "markov_best_n_states": markov_best,
        "hmm_best_n_states": hmm_best,
    }
