"""Simulate the 'Review Highlights' A/B test.

The product: on each restaurant page, a small module shows what reviewers praise
most (from the NLP pipeline's praise tags) plus one honest caveat (the top
complaint, e.g. "some reviewers mention late deliveries at peak hours").

Planted effects (the analysis notebooks do NOT read these constants; they have
to rediscover them from the data, the same way you would in real life):

* New users: the module helps them choose -> persistent lift in conversion.
* Returning users: they already know where they order; the module pushes the
  menu down and adds a caveat about places they like -> small persistent DROP.
* Both groups: a novelty bump that decays over ~3 days (shiny new thing).
* Platform and city tier: no real difference (so any "finding" there is noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EXPERIMENT, EXPERIMENT_SEED, EXPERIMENT_SESSIONS, EXPERIMENT_USERS

TRUE_EFFECTS = {  # relative change in per-session conversion probability
    "new": {"steady": 0.06, "novelty": 0.08},
    "returning": {"steady": -0.06, "novelty": 0.14},
}
NOVELTY_DECAY_DAYS = 3.0
VISIT_PROB = {"new": 0.10, "returning": 0.22}       # P(return visit) per later day
BASE_SESSION_CVR = {"new": 0.085, "returning": 0.08}  # returning also scales with pre-period orders
DOW_MULT = np.array([0.95, 0.93, 0.95, 1.0, 1.10, 1.12, 1.0])  # Mon..Sun (test starts on a Monday)


def simulate(cfg: dict = EXPERIMENT, seed: int = EXPERIMENT_SEED, effects: dict = TRUE_EFFECTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = cfg["planned_days"]
    per_day = cfg["daily_eligible_users"]
    n = per_day * days

    users = pd.DataFrame({
        "user_id": np.arange(n),
        "first_exposure_day": np.repeat(np.arange(days), per_day),
    })
    # deterministic-looking hash assignment; a fair coin in expectation
    users["variant"] = np.where(rng.random(n) < 0.5, "treatment", "control")
    users["user_type"] = np.where(rng.random(n) < 0.55, "new", "returning")
    users["platform"] = rng.choice(["ios", "android", "web"], size=n, p=[0.45, 0.35, 0.20])
    users["city_tier"] = rng.choice(["tier_1", "tier_2"], size=n, p=[0.6, 0.4])
    users["pre_period_orders"] = np.where(users.user_type == "returning", rng.poisson(2.2, n), 0)

    # ---- visits: day of first exposure + Bernoulli returns on later days -------
    day_grid = np.arange(days)
    q = users.user_type.map(VISIT_PROB).to_numpy()
    visits = rng.random((n, days)) < q[:, None]
    first = users.first_exposure_day.to_numpy()
    visits[np.arange(n), first] = True
    visits &= day_grid[None, :] >= first[:, None]
    uidx, day = np.nonzero(visits)

    s = pd.DataFrame({"user_id": uidx, "day": day})
    s = s.merge(users, on="user_id", how="left")
    s["days_since_exposure"] = s.day - s.first_exposure_day

    base = np.where(
        s.user_type == "new",
        BASE_SESSION_CVR["new"],
        BASE_SESSION_CVR["returning"] + 0.012 * np.minimum(s.pre_period_orders, 6),
    ) * DOW_MULT[s.day % 7]

    steady = s.user_type.map(lambda t: effects[t]["steady"]).to_numpy()
    novelty = s.user_type.map(lambda t: effects[t]["novelty"]).to_numpy()
    lift = steady + novelty * np.exp(-s.days_since_exposure / NOVELTY_DECAY_DAYS)
    p = base * np.where(s.variant == "treatment", 1 + lift, 1.0)
    s["ordered"] = (rng.random(len(s)) < np.clip(p, 0, 1)).astype(int)
    s["order_value"] = np.where(s.ordered == 1, np.round(rng.gamma(6.0, 9.5, len(s)), 2), 0.0)  # ~R$57 AOV
    s["refunded"] = np.where(s.ordered == 1, (rng.random(len(s)) < 0.035).astype(int), 0)

    sessions = s[["user_id", "day", "days_since_exposure", "variant", "user_type", "platform", "city_tier",
                  "ordered", "order_value", "refunded"]]

    # ---- user-level primary metric: ordered within 7 days of first exposure ----
    w = sessions[sessions.days_since_exposure < 7].groupby("user_id").agg(
        orders_7d=("ordered", "sum"), revenue_7d=("order_value", "sum"), refunds_7d=("refunded", "sum"))
    users = users.join(w, on="user_id")
    users["converted_7d"] = (users.orders_7d > 0).astype(int)
    users["full_7d_window"] = users.first_exposure_day <= cfg["exposure_cohort_last_day"]
    return users, sessions


def main() -> None:
    users, sessions = simulate()
    EXPERIMENT_USERS.parent.mkdir(parents=True, exist_ok=True)
    users.to_parquet(EXPERIMENT_USERS, index=False)
    sessions.to_parquet(EXPERIMENT_SESSIONS, index=False)
    print(f"{len(users):,} users, {len(sessions):,} sessions")


if __name__ == "__main__":
    main()
