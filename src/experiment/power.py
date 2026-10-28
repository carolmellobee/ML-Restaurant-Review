"""Power analysis for a two-arm test on a conversion rate.

Done BEFORE the test, from pre-period data. The output is a commitment:
sample size per arm, run length, and the MDE we accept. Anything decided after
looking at results (stopping early, changing the metric) breaks the guarantees.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def sample_size_two_proportions(p0: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8,
                                ratio: float = 1.0) -> int:
    """Users per arm (control) for a two-sided two-proportion z-test.

    Uses the unpooled-variance formula:
        n = (z_{1-a/2} * sqrt(p_bar(1-p_bar)(1+1/r)) + z_{1-b} * sqrt(p0(1-p0) + p1(1-p1)/r))^2 / d^2
    """
    p1 = p0 * (1 + mde_rel)
    d = p1 - p0
    p_bar = (p0 + ratio * p1) / (1 + ratio)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (za * math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) / ratio)) ** 2
    return math.ceil(num / d ** 2)


def mde_given_n(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative lift detectable with n users per arm (solved numerically)."""
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sample_size_two_proportions(p0, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def power_curve(p0: float, n_per_arm: int, lifts=np.linspace(0.0, 0.12, 49), alpha: float = 0.05) -> pd.DataFrame:
    za = norm.ppf(1 - alpha / 2)
    rows = []
    for L in lifts:
        p1 = p0 * (1 + L)
        se = math.sqrt(p0 * (1 - p0) / n_per_arm + p1 * (1 - p1) / n_per_arm)
        z = (p1 - p0) / se if se else 0
        rows.append({"relative_lift": L, "power": norm.cdf(z - za) + norm.cdf(-z - za)})
    return pd.DataFrame(rows)


def simulated_power(p0: float, mde_rel: float, n_per_arm: int, sims: int = 4000, alpha: float = 0.05,
                    seed: int = 0) -> float:
    """Monte Carlo check of the formula: never trust one formula you typed yourself."""
    rng = np.random.default_rng(seed)
    p1 = p0 * (1 + mde_rel)
    c = rng.binomial(n_per_arm, p0, sims) / n_per_arm
    t = rng.binomial(n_per_arm, p1, sims) / n_per_arm
    pool = (c + t) / 2
    z = (t - c) / np.sqrt(pool * (1 - pool) * 2 / n_per_arm)
    return float((np.abs(z) > norm.ppf(1 - alpha / 2)).mean())


def plan(cfg: dict) -> dict:
    n = sample_size_two_proportions(cfg["baseline_rate"], cfg["mde_relative"], cfg["alpha"], cfg["power"])
    per_day_per_arm = cfg["daily_eligible_users"] / 2
    # users only count once their 7-day window has closed, so add the 7 days of follow-up
    days_to_enroll = math.ceil(n / per_day_per_arm)
    return {
        "n_per_arm": n,
        "n_total": 2 * n,
        "days_to_enroll": days_to_enroll,
        "days_including_7d_followup": days_to_enroll + 7,
        "planned_days": cfg["planned_days"],
        "enrolled_per_arm_with_full_window": int(per_day_per_arm * (cfg["exposure_cohort_last_day"] + 1)),
    }
