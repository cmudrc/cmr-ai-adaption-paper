"""
Generate the full experimental design for SQLB simulation runs.

This script defines all global simulation parameters and constructs
the Cartesian product of:

    - random seed (replication group)
    - agents_average_initial_opinion
    - technology_success_rate

Each unique combination becomes one simulation configuration,
saved as a row in settings.csv.

Design structure:
    - seeds_num controls stochastic replications.
    - agents_average_initial_opinion_list spans the initial
      mean sentiment toward the technology.
    - technology_success_rate_list spans objective AI accuracy.
    - Each (seed, opinion, accuracy) triplet defines one model.

Naming convention:
    name = "{seed_index}_{opinion_index}_{accuracy_index}"

    This ensures:
        - deterministic mapping between settings and model outputs
        - compatibility with downstream ODE fitting and tradeoff analysis

Output:
    settings.csv with columns:
        name
        group                  (seed group identifier)
        seed
        agents_average_initial_opinion
        technology_success_rate

Experimental size:
    total_runs = seeds_num × len(opinion_list) × len(success_rate_list)

This file defines the parameter grid that later enables:
    - ODE reduction and k estimation
    - final adoption computation
    - tradeoff contour analysis
    - decision framework construction
"""

import csv
import numpy as np

from utils import BASE_DIR


# Steps
steps = 100

# Master seed
master_seed = 42
rng = np.random.default_rng(master_seed)

# Seeds
seeds_num = 5
seeds_list = rng.integers(low=0, high=2**32 - 1, size=seeds_num).tolist()

# Agents
agents_num = 100
agents_connection_probability = 0.5
agents_average_initial_opinion_list = [
    round(x, 1)
    for x in np.linspace(-1, 1, 21)
]
technology_use_cutoff_opinion = -0.2

# Teams
teams_num = 10
teams_connection_probability = 0.3

# Technology
technology_success_rate_list = [
    round(x, 1)
    for x in np.linspace(0, 1, 11)
]


if __name__ == "__main__":
    settings: list[dict] = []
    for s, seed in enumerate(seeds_list):
        for o, agents_average_initial_opinion in enumerate(agents_average_initial_opinion_list):
            for t, technology_success_rate in enumerate(technology_success_rate_list):      
                setting = {
                    "name": f"{s}_{o}_{t}",
                    "group": s, # groups represents all simulation with similar conditions but just different seed
                    "seed": seed,
                    "agents_average_initial_opinion": agents_average_initial_opinion,
                    "technology_success_rate": technology_success_rate,
                }
                settings.append(setting)

    fieldnames = settings[0].keys()

    settings_path = BASE_DIR / "settings_1.csv"
    with open(settings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(settings)

    print("Total runs:", len(settings))