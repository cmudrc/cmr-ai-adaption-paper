import os
from copy import deepcopy
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import trustdynamics as td

from utils import BASE_DIR, load_settings
from config import (
    steps,
    master_seed,
    agents_num,
    agents_connection_probability,
    teams_connection_probability,
    technology_use_cutoff_opinion,
)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- process-local globals (set by initializer) ---
_BASE_ORG_STRUCTURE = None


def _init_worker(base_org_structure):
    """
    Runs once per worker process.
    Stores an immutable base org structure in a process-global variable.
    """
    global _BASE_ORG_STRUCTURE
    _BASE_ORG_STRUCTURE = base_org_structure


def run_one(setting: dict) -> tuple[str, str]:
    """
    Run one model for one setting using the process-local base org structure.
    Returns (name, status).
    """
    global _BASE_ORG_STRUCTURE
    if _BASE_ORG_STRUCTURE is None:
        raise RuntimeError("Worker not initialized with base org structure.")

    name = str(setting["name"])
    path = MODELS_DIR / f"{name}.json"
    if path.exists():
        return name, "skipped (exists)"

    # copy base structure for this run
    organization = deepcopy(_BASE_ORG_STRUCTURE)

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
    model.run(steps=steps, show_progress=False)
    model.save(path=path)

    return name, "saved"


def build_base_org_structure(*, teams_num: int) -> object:
    """
    Build the organization structure for a given teams_num.
    Keep it "pre-initialized" (no seed-dependent initialization),
    so each run can initialize independently.
    """
    return td.organization.generate_random_organization_structure(
        teams_num=int(teams_num),
        agents_num=int(agents_num),
        agents_connection_probability_inside_team=float(agents_connection_probability),
        teams_connection_probability=float(teams_connection_probability),
        technology_use_cutoff_opinion=float(technology_use_cutoff_opinion),
        seed=int(master_seed),  # same base topology seed for comparability
    )


def group_settings_by_teams_num(settings: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for st in settings:
        tn = int(st["teams_num"])
        grouped[tn].append(st)
    return grouped


def main():
    settings = load_settings()
    grouped = group_settings_by_teams_num(settings)

    # leave 2 cores free
    max_workers = max(1, (os.cpu_count() or 2) - 2)

    total_all = len(settings)
    done_all = 0

    for teams_num, group in sorted(grouped.items(), key=lambda x: x[0]):
        base_org = build_base_org_structure(teams_num=teams_num)

        print(f"\n=== teams_num={teams_num} | runs={len(group)} ===")

        # New pool per teams_num so each pool has a single initializer value
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(base_org,),
        ) as ex:
            futures = [ex.submit(run_one, st) for st in group]

            for fut in as_completed(futures):
                name, status = fut.result()
                done_all += 1
                print(f"[{done_all}/{total_all}] {name}: {status}")


if __name__ == "__main__":
    main()