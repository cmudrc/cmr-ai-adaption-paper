import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import BASE_DIR


def _load_states_df(base_dir: Path, model_name: str) -> pd.DataFrame:
    csv_path = base_dir / "states" / f"{model_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"State CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    # Keep a stable state order
    df = df.sort_values("t").reset_index(drop=True)
    return df


def _load_Q_npz(base_dir: Path, model_name: str):
    npz_path = base_dir / "odes" / f"{model_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"ODE fit file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    Q = data["Q"].astype(float)
    dt = float(data["dt"][0]) if "dt" in data else 1.0
    state_order = data["state_order"].tolist() if "state_order" in data else ["S", "Q", "L", "B"]

    return Q, dt, state_order


def _simulate_ode_fractions(
    Q: np.ndarray,
    x0: np.ndarray,
    T: int,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Forward simulate fractions with explicit Euler:
        x_{t+1} = x_t + dt * x_t Q
    """
    x = np.zeros((T, 4), dtype=float)
    x[0] = x0

    for t in range(T - 1):
        x[t + 1] = x[t] + dt * (x[t] @ Q)

        # numerical safety: clip and renormalize
        x[t + 1] = np.clip(x[t + 1], 0.0, 1.0)
        s = x[t + 1].sum()
        if s > 0:
            x[t + 1] /= s

    return x


def plot_sqlb_states(base_dir: Path, model_name: str, *, show: bool = True, save_path: str | Path | None = None):
    df = _load_states_df(base_dir, model_name)

    plt.figure()
    plt.plot(df["t"], df["ratio_S"], label="S")
    plt.plot(df["t"], df["ratio_Q"], label="Q")
    plt.plot(df["t"], df["ratio_L"], label="L")
    plt.plot(df["t"], df["ratio_B"], label="B")

    plt.xlabel("Time step (t)")
    plt.ylabel("Agent ratio")
    plt.title(f"SQLB state ratios — {model_name}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return df


def plot_sqlb_states_vs_ode(
    base_dir: Path,
    model_name: str,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Overlay ABM state ratios (solid) vs ODE simulated ratios (dashed),
    with consistent colors per state.
    """
    df = _load_states_df(base_dir, model_name)
    Q, dt_fit, state_order = _load_Q_npz(base_dir, model_name)

    # Ensure ordering matches (expects S,Q,L,B)
    desired = ["S", "Q", "L", "B"]
    if list(state_order) != desired:
        # reorder Q into desired order
        idx = {s: i for i, s in enumerate(state_order)}
        perm = [idx[s] for s in desired]
        Q = Q[np.ix_(perm, perm)]
        state_order = desired

    T = len(df)
    t = df["t"].to_numpy()

    x0 = np.array([df.loc[0, "ratio_S"], df.loc[0, "ratio_Q"], df.loc[0, "ratio_L"], df.loc[0, "ratio_B"]], dtype=float)
    ode = _simulate_ode_fractions(Q, x0, T, dt=dt_fit)

    # --- consistent colors: plot ABM first, reuse those colors for ODE ---
    fig = plt.figure()
    ax = plt.gca()

    line_S = ax.plot(t, df["ratio_S"], label="S (ABM)")[0]
    line_Q = ax.plot(t, df["ratio_Q"], label="Q (ABM)")[0]
    line_L = ax.plot(t, df["ratio_L"], label="L (ABM)")[0]
    line_B = ax.plot(t, df["ratio_B"], label="B (ABM)")[0]

    colors = {
        "S": line_S.get_color(),
        "Q": line_Q.get_color(),
        "L": line_L.get_color(),
        "B": line_B.get_color(),
    }

    ax.plot(t, ode[:, 0], linestyle="--", color=colors["S"], label="S (ODE)")
    ax.plot(t, ode[:, 1], linestyle="--", color=colors["Q"], label="Q (ODE)")
    ax.plot(t, ode[:, 2], linestyle="--", color=colors["L"], label="L (ODE)")
    ax.plot(t, ode[:, 3], linestyle="--", color=colors["B"], label="B (ODE)")

    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Agent ratio")
    ax.set_title(f"SQLB — ABM (solid) vs ODE (dashed) — {model_name}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return df, ode, Q, dt_fit


# Example:
plot_sqlb_states_vs_ode(BASE_DIR, "0_9_8")

#plot_sqlb_states(BASE_DIR, "0_9_8")