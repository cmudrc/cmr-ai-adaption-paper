"""
Useful stuff for the rest of pipeline.
"""

import csv
from pathlib import Path
import trustdynamics as td


BASE_DIR = Path(__file__).resolve().parent

def get_all_model_names(base_dir: Path) -> list[str]:
    with open(base_dir / "settings.csv" , newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["name"] for row in reader]
    
#models_names = get_all_model_names(BASE_DIR)

def get_model(base_dir: Path, name: str) -> td.Model:
    path = base_dir / "models"
    path = path / f"{name}.json"
    model = td.Model.load(path)
    return model

#model = get_model(BASE_DIR, '0_0_0')

def load_settings(base_dir: Path):
    settings_path = BASE_DIR / "settings.csv"
    with settings_path.open(mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        settings = [dict(row) for row in reader]
    return settings