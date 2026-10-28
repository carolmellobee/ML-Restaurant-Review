"""Rerun the experiment simulation + key analyses over many seeds.

A single simulated dataset is one draw. If the conclusion only holds for the seed
in config.py, the project is a magic trick, not an analysis. This script shows how
often each headline result holds across 20 independent draws.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from src.config import DATA_PROCESSED  # noqa: E402
from src.experiment.analysis import compare_proportions, interaction_test, long_run_effect  # noqa: E402
from src.experiment.simulate import simulate  # noqa: E402


def one(seed: int) -> dict:
    users, sessions = simulate(seed=seed)
    u = users[users.full_7d_window]
    head = compare_proportions(u, "converted_7d")
    ret7 = compare_proportions(u[u.user_type == "returning"], "converted_7d")
    lr = long_run_effect(sessions, by="user_type").set_index("level")
    lr_all = long_run_effect(sessions).iloc[0]
    city = interaction_test(u, "city_tier").pvalues["treat:C(city_tier)[T.tier_2]"]
    return {
        "seed": seed,
        "headline_7d_lift": head["rel_lift"], "headline_p": head["p_value"],
        "returning_7d_lift": ret7["rel_lift"],
        "longrun_new": lr.loc["new", "rel_lift"], "longrun_returning": lr.loc["returning", "rel_lift"],
        "longrun_returning_ci_high": lr.loc["returning", "rel_ci_high"],
        "longrun_all": lr_all["rel_lift"],
        "city_tier_interaction_p": city,   # no true city effect exists: this is the false-positive check
    }


def main(n: int = 20) -> pd.DataFrame:
    res = pd.DataFrame([one(s) for s in range(1, n + 1)])
    summary = pd.Series({
        "headline significant (p<0.05)": (res.headline_p < 0.05).mean(),
        "returning users look positive on 7-day metric": (res.returning_7d_lift > 0).mean(),
        "returning long-run CI entirely below 0": (res.longrun_returning_ci_high < 0).mean(),
        "full-rollout long-run effect <= 0": (res.longrun_all <= 0).mean(),
        "city-tier interaction p<0.05 (pure noise)": (res.city_tier_interaction_p < 0.05).mean(),
    })
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    res.to_csv(DATA_PROCESSED / "seed_robustness.csv", index=False)
    print(res.round(3).to_string(index=False))
    print("\nShare of seeds where:")
    print(summary.map("{:.0%}".format).to_string())
    return res


if __name__ == "__main__":
    main()
