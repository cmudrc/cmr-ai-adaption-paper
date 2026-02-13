"""
Render the decision surface: AI accuracy vs average initial opinion.

This module visualizes the organizational adoption response surface
derived from ODE-reduced SQLB dynamics. It plots final predicted
adoption as a function of two controllable levers:

    1. technology_success_rate  (AI accuracy)
    2. agents_average_initial_opinion  (baseline organizational sentiment)

The adoption metric used is:

    final_adaption = Q(T) + L(T)

where Q and L are the fractions of agents in adoption states at the
end of the simulation horizon T, predicted by the fitted ODE surrogate.

Workflow:
    - Load final_adaption.csv (generated from ODE-based adoption calculation).
    - Pivot the experimental grid into a 2D matrix:
            rows    → average initial opinion
            columns → AI accuracy
            values  → final adoption
    - Plot a filled contour (heatmap) of adoption across the design space.

Interpretation (decision framework):
    - The surface shows how improvements in AI accuracy can compensate
      for lower initial sentiment, and vice versa.
    - Regions of high adoption indicate feasible zones for successful rollout.
    - If iso-adoption contours are enabled, each contour represents a
      constant adoption target (a decision boundary).

Assumptions:
    - The experimental design forms a complete grid (no missing combinations).
    - Adoption values lie in [0, 1]; color scaling is constrained accordingly.
    - Each grid point represents an independent simulation configuration.

Output:
    - Displays the figure interactively.
    - Optionally saves a publication-ready figure to disk.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import BASE_DIR


def plot_tradeoff_iso_adoption(
    base_dir: Path,
    *,
    csv_name: str = "final_adaption.csv",
    levels: list[float] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Plot the tradeoff between technology_success_rate (x) and agents_average_initial_opinion (y),
    using final_adaption as the outcome.

    Expects base_dir/final_adaption.csv with columns:
        name, agents_average_initial_opinion, technology_success_rate, final_adaption
    """
    if levels is None:
        levels = [0.2, 0.4, 0.6, 0.8]

    path = base_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path)

    required = {"agents_average_initial_opinion", "technology_success_rate", "final_adaption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    # Ensure numeric
    df["agents_average_initial_opinion"] = pd.to_numeric(df["agents_average_initial_opinion"], errors="raise")
    df["technology_success_rate"] = pd.to_numeric(df["technology_success_rate"], errors="raise")
    df["final_adaption"] = pd.to_numeric(df["final_adaption"], errors="raise")

    # Pivot to grid (rows: opinion, cols: success_rate)
    grid = df.pivot_table(
        index="agents_average_initial_opinion",
        columns="technology_success_rate",
        values="final_adaption",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    if grid.isna().any().any():
        # Contours require a complete grid; fail loudly of not. In that case, interpolation is needed.
        nan_locs = np.argwhere(grid.isna().to_numpy())
        raise ValueError(
            f"Grid has missing cells (NaNs). Example missing at {nan_locs[:5].tolist()} "
            f"after pivot. Ensure you ran all combinations or enable interpolation."
        )

    x_vals = grid.columns.to_numpy(dtype=float)  # technology_success_rate
    y_vals = grid.index.to_numpy(dtype=float)    # agents_average_initial_opinion
    Z = grid.to_numpy(dtype=float)

    X, Y = np.meshgrid(x_vals, y_vals)

    plt.figure()
    color_levels = np.linspace(0.0, 1.0, 11)
    # filled contour for adoption surface
    cf = plt.contourf(
        X, Y, Z,
        levels=color_levels,
        vmin=0.0,
        vmax=1.0
    )

    # iso-adoption curves
    #cs = plt.contour(X, Y, Z, levels=levels)
    #plt.clabel(cs, inline=True, fontsize=9, fmt="A=%.1f")

    plt.xlabel("AI Accuracy")
    plt.ylabel("Average Initial Opinion")
    plt.title("Tradeoff: AI Accuracy vs Average Initial Opinion")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(cf)
    cbar.set_label("Final Adoption")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return grid


if __name__ == "__main__":
    # Example:
    plot_tradeoff_iso_adoption(
        BASE_DIR,
        levels=[0.2, 0.4, 0.6, 0.8],
        save_path=Path(BASE_DIR) / "figures" / "final_adoption.png",
    )