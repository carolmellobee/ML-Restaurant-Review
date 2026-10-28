"""Central configuration: paths, seeds, and the experiment's pre-registered design.

Everything that a reviewer would want to audit (alpha, power, MDE, segments we
committed to checking) lives here so it cannot silently drift between notebooks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

REVIEWS_CSV = DATA_RAW / "reviews.csv"
REVIEWS_ENRICHED = DATA_PROCESSED / "reviews_enriched.csv"
EXPERIMENT_SESSIONS = DATA_RAW / "experiment_sessions.parquet"
EXPERIMENT_USERS = DATA_RAW / "experiment_users.parquet"

SEED = 42
# Seed for the simulated experiment. scripts/seed_robustness.py reruns the whole
# analysis over 20 seeds so you can see this one is typical, not cherry-picked.
EXPERIMENT_SEED = 2

# ---------------------------------------------------------------------------
# Review generation
# ---------------------------------------------------------------------------
N_REVIEWS = 3000
CUISINES = ["Italian", "Japanese", "Brazilian", "Mexican", "Burgers", "Indian", "Vegan", "Pizza"]

# ---------------------------------------------------------------------------
# Experiment: "Review Highlights" module on restaurant pages
# ---------------------------------------------------------------------------
EXPERIMENT = {
    "name": "review_highlights_v1",
    "hypothesis": (
        "Showing NLP-derived review highlights (top praised topics + one honest caveat) "
        "on restaurant pages increases the share of users who place an order within "
        "7 days of first exposure."
    ),
    "unit": "user",
    "primary_metric": "converted_7d",          # user ordered >= 1 time within 7 days of first exposure
    "guardrails": ["orders_per_converter_7d", "refund_rate"],
    "alpha": 0.05,                            # two-sided
    "power": 0.80,
    "baseline_rate": 0.18,                    # from 4 weeks of pre-period data
    "mde_relative": 0.05,                     # smallest lift worth shipping (+5% relative)
    "daily_eligible_users": 6000,             # users entering the test per day (both arms)
    "planned_days": 28,                       # 4 full weeks: weekly seasonality + novelty read
    "exposure_cohort_last_day": 21,           # users first exposed on day <= 21 get a full 7-day window (days 0-27)
    "interim_looks": 4,                       # pre-registered weekly looks (O'Brien-Fleming)
    "preregistered_segments": ["user_type", "platform", "city_tier"],
}
