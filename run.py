import os
import csv
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import trustdynamics as td

from utils import BASE_DIR
from config import (
    steps,
    master_seed,
    agents_num,
    teams_num,
    agents_connection_probability,
    teams_connection_probability,
    technology_use_cutoff_opinion,
)


# ---- Load settings ----
settings_path = BASE_DIR / "settings.csv"
with settings_path.open(mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    settings = [dict(row) for row in reader]

models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# ---- Build shared organization structure ONCE in parent ----
ORG_STRUCTURE = td.organization.generate_random_organization_structure(
    teams_num=teams_num,
    agents_num=agents_num,
    agents_connection_probability_inside_team=agents_connection_probability,
    teams_connection_probability=teams_connection_probability,
    technology_use_cutoff_opinion=technology_use_cutoff_opinion,
    seed=master_seed,
)

def run_one(setting: dict) -> tuple[str, str]:
    """
    Build, run, save one model. Returns (name, status).
    """
    name = str(setting["name"])
    path = models_dir / f"{name}.json"
    if path.exists():
        return name, "skipped (exists)"

    # Each process gets its own copy
    organization = deepcopy(ORG_STRUCTURE)
    seed = int(setting["seed"])

    organization.initialize(
        agents_average_initial_opinion=float(setting["agents_average_initial_opinion"]),
        seed=seed,
    )

    technology = td.Technology(
        success_rate=float(setting["technology_success_rate"]),
        seed=seed,
    )

    model = td.Model(organization, technology)
    model.run(
        steps=steps,
        show_progress=False, # tqdm doesn't work in parallel
    )
    model.save(path=path)

    return name, "saved"


if __name__ == "__main__":

    max_workers = max(1, os.cpu_count() - 2)  # leave 2 cores free

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, st) for st in settings]

        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{total}] {name}: {status}")