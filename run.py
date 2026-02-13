import csv
from copy import deepcopy
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

# Load settings
settings_path = BASE_DIR / "settings.csv"
with settings_path.open(mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    settings = [dict(row) for row in reader]
len_settings = len(settings)
#print(settings[0])

# Create organization (same across different runs)
organization_structure = td.organization.generate_random_organization_structure(
    teams_num=teams_num,
    agents_num=agents_num,
    agents_connection_probability_inside_team=agents_connection_probability,
    teams_connection_probability=teams_connection_probability,
    seed=master_seed
)

# Create models, run and save them
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)
for s, setting in enumerate(settings):

    path = models_dir / f"{str(setting["name"])}.json"
    if path.exists(): # skipping existing ones
        continue
    
    # Initialize organization
    organization = deepcopy(organization_structure)
    organization.initialize(
        agents_average_initial_opinion=float(setting["agents_average_initial_opinion"]),
        technology_use_cutoff_opinion=technology_use_cutoff_opinion,
        seed=int(setting["seed"])
    )

    # Create technology
    technology = td.Technology(
        success_rate=float(setting["technology_success_rate"]),
        seed=int(setting["seed"])
    )

    # Create model
    model = td.Model(organization, technology)

    # Run model
    model.run(steps=steps)
    print(f"Progress: {s} / {len_settings}")

    # Save
    model.save(path=path)
