"""
Render the decision surface: AI accuracy vs average initial opinion.

This module visualizes the organizational adoption response surface
derived from ODE-reduced SQLB dynamics. It plots final predicted
adoption as a function of two controllable levers:

    1. technology_success_rate  (AI accuracy)
    2. agents_average_initial_opinion  (baseline organizational sentiment)

The adoption metric used is:

    final_adoption = Q(T) + L(T)

where Q and L are the fractions of agents in adoption states at the
end of the simulation horizon T, predicted by the fitted ODE surrogate.

Workflow:
    - Load final_adoption.csv (generated from ODE-based adoption calculation).
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
from matplotlib.ticker import MultipleLocator

from utils import BASE_DIR, load_settings, get_all_model_names
from config import teams_num_list


def plot_tradeoff_iso_adoption(
    base_dir: Path,
    *,
    csv_name: str = "final_adoption.csv",
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Plot the tradeoff between technology_success_rate (x) and agents_average_initial_opinion (y),
    using final_adoption as the outcome.

    Expects base_dir/final_adoption.csv with columns:
        name, final_adoption

    Uses settings.csv (via load_settings) to attach:
        teams_num, agents_average_initial_opinion, technology_success_rate

    Produces one figure per teams_num in config.teams_num_list.
    """
    path = base_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path)

    required = {"name", "final_adoption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    # Ensure numeric adoption
    df["final_adoption"] = pd.to_numeric(df["final_adoption"], errors="raise")

    # ---- load settings and join on name ----
    settings = pd.DataFrame(load_settings())

    required_settings = {"name", "teams_num", "agents_average_initial_opinion", "technology_success_rate"}
    missing_settings = required_settings - set(settings.columns)
    if missing_settings:
        raise ValueError(f"Missing columns in settings.csv: {sorted(missing_settings)}")

    settings["teams_num"] = pd.to_numeric(settings["teams_num"], errors="raise").astype(int)
    settings["agents_average_initial_opinion"] = pd.to_numeric(settings["agents_average_initial_opinion"], errors="raise")
    settings["technology_success_rate"] = pd.to_numeric(settings["technology_success_rate"], errors="raise")

    df = df.merge(
        settings[["name", "teams_num", "agents_average_initial_opinion", "technology_success_rate"]],
        on="name",
        how="inner",
    )

    for col in ["agents_average_initial_opinion", "technology_success_rate"]:
        x = f"{col}_x"
        y = f"{col}_y"
        if x in df.columns and y in df.columns:
            df[col] = df[y]
            df = df.drop(columns=[x, y])
        elif x in df.columns:
            df[col] = df[x]
            df = df.drop(columns=[x])
        elif y in df.columns:
            df[col] = df[y]
            df = df.drop(columns=[y])

    if df.empty:
        raise ValueError("After merging final_adoption.csv with settings.csv, no rows remained. Check name matching.")

    # ---- saving behavior ----
    out_dir: Path | None = None
    out_base: Path | None = None
    if save_path is not None:
        sp = Path(save_path)
        if sp.suffix:
            out_base = sp
            out_base.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = sp
            out_dir.mkdir(parents=True, exist_ok=True)

    grids = {}

    for tn in teams_num_list:
        # robust selection using your helper
        names_tn = set(get_all_model_names(teams_num=tn))
        df_tn = df[df["name"].isin(names_tn)].copy()

        if df_tn.empty:
            print(f"Skipping teams_num={tn}: no matching rows found.")
            continue

        # Pivot to grid (rows: opinion, cols: success_rate)
        grid = df_tn.pivot_table(
            index="agents_average_initial_opinion",
            columns="technology_success_rate",
            values="final_adoption",
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)

        if grid.isna().any().any():
            nan_locs = np.argwhere(grid.isna().to_numpy())
            raise ValueError(
                f"teams_num={tn}: Grid has missing cells (NaNs). Example missing at {nan_locs[:5].tolist()} "
                f"after pivot. Ensure you ran all combinations or enable interpolation."
            )

        x_vals = grid.columns.to_numpy(dtype=float)
        y_vals = grid.index.to_numpy(dtype=float)
        Z = grid.to_numpy(dtype=float)
        X, Y = np.meshgrid(x_vals, y_vals)

        plt.figure()
        #color_levels = np.linspace(0.0, 1.0, 11)
        color_levels = np.array([0.0, 0.2, 0.8, 1.0])
        cf = plt.contourf(X, Y, Z, levels=color_levels, vmin=0.0, vmax=1.0)

        plt.xlabel("AI Accuracy", labelpad=-10)
        plt.ylabel("Average Initial Opinion", labelpad=-30)
        #plt.title(f"Tradeoff: AI Accuracy vs Average Initial Opinion (teams_num={tn})")
        plt.title(f"Number of teams: {tn}")
        
        ax = plt.gca()

        # ---- major ticks (semantic labels) ----
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Low", "High"])

        ax.set_yticks([-1.0, 1.0])
        ax.set_yticklabels(["Negative", "Positive"])

        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # ---- minor tick spacing (grid resolution control) ----
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))   # every 0.1 on x
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))   # every 0.2 on y

        # ---- grid ----
        #ax.grid(which="minor", alpha=0.2)
        #ax.grid(which="major", alpha=0.4)
        ax.axhline(
            y=0.0,
            color='gray',
            #linestyle='--',
            linewidth=1.2,
            alpha=0.5
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        # optional: fix limits
        #plt.xlim(0.0, 1.0)
        #plt.ylim(-1.0, 1.0)

        cbar = plt.colorbar(cf)
        # ---- midpoints of each regime ----
        region_midpoints = [
            (0.0 + 0.2) / 2,   # 0.1
            (0.2 + 0.8) / 2,   # 0.5
            (0.8 + 1.0) / 2    # 0.9
        ]

        cbar.set_ticks(region_midpoints)
        cbar.set_ticklabels([
            "Low",
            "Partial",
            "High"
        ])

        cbar.set_label("Adoption Regime")

        # decide output filename for this tn
        out = None
        if out_dir is not None:
            out = out_dir / f"final_adoption_teams_{tn}.png"
        elif out_base is not None:
            out = out_base.with_name(f"{out_base.stem}_teams_{tn}{out_base.suffix}")

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, bbox_inches="tight", dpi=200)

        if show:
            plt.show()
        else:
            plt.close()

        grids[tn] = grid

    return grids


if __name__ == "__main__":
    plot_tradeoff_iso_adoption(
        BASE_DIR,
        show=False,
        save_path=Path(BASE_DIR) / "figures",
    )