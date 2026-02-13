import csv
from config import technology_use_cutoff_opinion
import trustdynamics as td

from config import steps, BASE_DIR
from utils import get_all_model_names, get_model


LOUDNESS_BOUND = 0.5

def sqlb_state(
    opinion: float,
    loudness: float,
    *,
    loudness_bound: float,
) -> str:
    """
    Classify agent into SQLB state under hierarchical logic.

    B: opinion < technology_use_cutoff_opinion
    S: technology_use_cutoff_opinion <= opinion < 0
    Q: opinion >= 0 and loudness < loudness_bound
    L: opinion >= 0 and loudness >= loudness_bound
    """
    # ---- Belief layer ----
    if opinion < technology_use_cutoff_opinion:
        return "B"
    if opinion < 0.0:
        return "S"

    # ---- Activation layer (only for advocates) ----
    if loudness < loudness_bound:
        return "Q"
    else:
        return "L"

def get_opinion(model: td.Model, agent: int | str, history_index: int):
    return model.organization.get_agent_opinion(agent, history_index)

def get_inbound_trust(model: td.Model, agent: int | str, history_index: int):
    """
    Inbound trust mass: sum of trust from neighbors -> agent.
    """
    neighbors = list(model.organization.agents_connected_to(agent))
    inbound = 0.0
    for j in neighbors:
        inbound += float(model.organization.get_agent_trust(j, agent, history_index=history_index))
    return inbound

def get_state(model: td.Model, agent: int | str, history_index: int, loudness_bound: float):
    opinion = get_opinion(model, agent, history_index)
    inbound_trust = get_inbound_trust(model, agent, history_index)
    loudness = opinion * inbound_trust
    return sqlb_state(opinion, loudness, loudness_bound=loudness_bound)


if __name__ == "__main__":
    states_dir = BASE_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    model_names = get_all_model_names(BASE_DIR)
    for model_name in model_names:
        model = get_model(BASE_DIR, model_name)
        agents = list(model.organization.all_agent_ids)
        n_agents = len(agents)
        if n_agents == 0:
            continue

        out_path = states_dir / f"{model_name}.csv"

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"],
            )
            writer.writeheader()

            for t in range(steps):
                counts = {"S": 0, "Q": 0, "L": 0, "B": 0}
                for agent in agents:
                    st = get_state(model, agent, t, loudness_bound=LOUDNESS_BOUND)
                    counts[st] += 1
                writer.writerow(
                    {
                        "t": t,
                        "ratio_S": counts["S"] / n_agents,
                        "ratio_Q": counts["Q"] / n_agents,
                        "ratio_L": counts["L"] / n_agents,
                        "ratio_B": counts["B"] / n_agents,
                    }
                )

        print(f"Wrote {out_path}")