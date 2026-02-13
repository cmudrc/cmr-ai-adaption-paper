# regress_k.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ALLOWED_STATE_CHANGES = {
    "S": ["S", "Q", "L"],
    "Q": ["Q", "L", "B"],
    "L": ["L", "Q", "B"],
    "B": ["B", "S"],
}
STATE_ORDER = list(ALLOWED_STATE_CHANGES.keys())


def allowed_transitions(state_order: List[str] = STATE_ORDER) -> List[str]:
    out = []
    for s in state_order:
        for t in ALLOWED_STATE_CHANGES[s]:
            if t != s:
                out.append(f"{s}->{t}")
    # de-dup preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def load_settings(base_dir: Path) -> pd.DataFrame:
    """
    Reads:
        base_dir/settings.csv
    Expected header:
        name,group,seed,agents_average_initial_opinion,technology_success_rate
    """
    path = base_dir / "settings.csv"
    if not path.exists():
        raise FileNotFoundError(f"settings.csv not found: {path}")

    df = pd.read_csv(path)

    required = {"name", "agents_average_initial_opinion", "technology_success_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in settings.csv: {sorted(missing)}")

    # enforce numeric
    df["agents_average_initial_opinion"] = pd.to_numeric(df["agents_average_initial_opinion"], errors="raise")
    df["technology_success_rate"] = pd.to_numeric(df["technology_success_rate"], errors="raise")

    return df


def load_k_npz(base_dir: Path, model_name: str) -> Dict[str, float]:
    """
    Loads one-file-per-model ODE fit:
        base_dir/odes/{model_name}.npz
    Returns:
        dict mapping transition string to rate
    """
    npz_path = base_dir / "odes" / f"{model_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"ODE file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    transitions = data["transitions"].tolist()
    rates = data["rates"].astype(float)
    return {tr: float(rt) for tr, rt in zip(transitions, rates)}


def build_regression_table(base_dir: Path) -> pd.DataFrame:
    """
    Merge settings with fitted k's:
        rows = models
        cols = [name, agents_average_initial_opinion, technology_success_rate, k_* ...]
    """
    settings = load_settings(base_dir)

    trs = allowed_transitions()
    rows = []

    for _, r in settings.iterrows():
        model_name = str(r["name"])
        try:
            k = load_k_npz(base_dir, model_name)
        except FileNotFoundError:
            # Skip models that don't have an ODE fit yet
            continue

        row = {
            "name": model_name,
            "agents_average_initial_opinion": float(r["agents_average_initial_opinion"]),
            "technology_success_rate": float(r["technology_success_rate"]),
        }
        for tr in trs:
            row[f"k_{tr}"] = float(k.get(tr, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows merged. Check that settings.csv names match odes/{name}.npz files.")

    return df


@dataclass(frozen=True)
class RegressionResult:
    target: str
    coef: Dict[str, float]   # includes intercept
    rmse: float
    r2: float


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Solve min ||Xb - y||_2
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # (X^T X + alpha I) b = X^T y
    p = X.shape[1]
    I = np.eye(p)
    b = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
    return b


def regress_k(
    base_dir: Path,
    *,
    ridge_alpha: float | None = None,
) -> Tuple[pd.DataFrame, List[RegressionResult]]:
    """
    Fit one regression per k target:
        k ~ intercept + a*agents_average_initial_opinion + b*technology_success_rate

    Returns:
        (dataframe_used, list_of_results)
    """
    df = build_regression_table(base_dir)

    features = ["agents_average_initial_opinion", "technology_success_rate"]
    X = df[features].to_numpy(dtype=float)

    # add intercept
    X_design = np.column_stack([np.ones(len(df)), X])
    feature_names = ["intercept"] + features

    results: List[RegressionResult] = []

    targets = [c for c in df.columns if c.startswith("k_")]
    for target in targets:
        y = df[target].to_numpy(dtype=float)

        if ridge_alpha is None:
            b = _ols(X_design, y)
        else:
            b = _ridge(X_design, y, float(ridge_alpha))

        yhat = X_design @ b
        resid = y - yhat
        rmse = float(np.sqrt(np.mean(resid**2)))

        # R^2
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        coef = {name: float(val) for name, val in zip(feature_names, b)}
        results.append(RegressionResult(target=target, coef=coef, rmse=rmse, r2=r2))

    return df, results


def save_regression_outputs(base_dir: Path, df: pd.DataFrame, results: List[RegressionResult]) -> Path:
    """
    Saves:
        base_dir/regression/regression_table.csv
        base_dir/regression/regression_coeffs.csv
    """
    out_dir = base_dir / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "regression_table.csv", index=False)

    rows = []
    for r in results:
        row = {"target": r.target, "rmse": r.rmse, "r2": r.r2}
        row.update(r.coef)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "regression_coeffs.csv", index=False)

    return out_dir


if __name__ == "__main__":
    # Example usage:
    BASE_DIR = Path(__file__).resolve().parent  # or import from utils if you prefer
    df, results = regress_k(BASE_DIR, ridge_alpha=None)  # set ridge_alpha=1e-3 if noisy/collinear
    out_dir = save_regression_outputs(BASE_DIR, df, results)
    print(f"Saved regression outputs to: {out_dir}")
    # Print a quick summary
    for r in sorted(results, key=lambda x: (-np.nan_to_num(x.r2, nan=-1.0), x.rmse))[:5]:
        print(r.target, "R2=", r.r2, "RMSE=", r.rmse, "coef=", r.coef)