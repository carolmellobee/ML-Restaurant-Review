"""The three traps: peeking, novelty, and a segment where the effect reverses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Trap 1: peeking
# ---------------------------------------------------------------------------
def _cum_z(rng, p, n_per_look, looks, sims):
    """Cumulative z-statistics for `sims` A/A tests observed at `looks` points."""
    c = rng.binomial(n_per_look, p, (sims, looks)).cumsum(1)
    t = rng.binomial(n_per_look, p, (sims, looks)).cumsum(1)
    n = n_per_look * np.arange(1, looks + 1)
    pp = (c + t) / (2 * n)
    se = np.sqrt(pp * (1 - pp) * 2 / n)
    return (t - c) / n / se


def aa_peeking_simulation(p: float = 0.18, n_per_day_per_arm: int = 3000, days: int = 28, sims: int = 4000,
                          alpha: float = 0.05, seed: int = 7) -> dict:
    """A/A tests: there is NO effect, so every 'win' is a false positive."""
    rng = np.random.default_rng(seed)
    z = _cum_z(rng, p, n_per_day_per_arm, days, sims)
    crit = norm.ppf(1 - alpha / 2)
    hit = np.abs(z) > crit
    any_hit = hit.any(1)
    first_day = np.where(any_hit, hit.argmax(1) + 1, np.nan)
    fpr_by_n_peeks = {k: float(hit[:, np.linspace(0, days - 1, k).round().astype(int)].any(1).mean())
                      for k in [1, 2, 4, 7, 14, 28]}
    return {"fpr_single_look": float(hit[:, -1].mean()), "fpr_daily_peeking": float(any_hit.mean()),
            "fpr_by_number_of_peeks": fpr_by_n_peeks, "first_false_win_day": first_day, "z_paths": z[:200]}


def obf_constant(looks: int, alpha: float = 0.05, sims: int = 200_000, seed: int = 11) -> float:
    """O'Brien-Fleming: reject at look k if |Z_k| > c * sqrt(K / k). Solve for c by simulation."""
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((sims, looks))
    z = inc.cumsum(1) / np.sqrt(np.arange(1, looks + 1))
    scale = np.sqrt(looks / np.arange(1, looks + 1))
    # max over looks of |Z_k| / scale_k; c is its (1-alpha) quantile
    return float(np.quantile((np.abs(z) / scale).max(1), 1 - alpha))


def obf_fpr_check(p: float = 0.18, n_per_week_per_arm: int = 21000, looks: int = 4, sims: int = 4000,
                  alpha: float = 0.05, seed: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    c = obf_constant(looks, alpha)
    bounds = c * np.sqrt(looks / np.arange(1, looks + 1))
    z = _cum_z(rng, p, n_per_week_per_arm, looks, sims)
    return {"obf_c": c, "z_boundaries": bounds, "fpr": float((np.abs(z) > bounds).any(1).mean()),
            "nominal_alpha_per_look": 2 * (1 - norm.cdf(bounds))}


def observed_peeking_path(users: pd.DataFrame, sessions: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    """What the dashboard showed each morning: cumulative 'converted so far' among users
    exposed so far (each user observed for at most 7 days)."""
    first_order = (sessions[(sessions.ordered == 1) & (sessions.days_since_exposure < 7)]
                   .groupby("user_id").day.min())
    u = users[["user_id", "variant", "first_exposure_day"]].copy()
    u["first_order_day"] = u.user_id.map(first_order)
    rows = []
    for d in range(days):
        seen = u[u.first_exposure_day <= d]
        conv = (seen.first_order_day <= d).astype(int)
        g = conv.groupby(seen.variant).agg(["sum", "count"])
        pc, pt = g.loc["control", "sum"] / g.loc["control", "count"], g.loc["treatment", "sum"] / g.loc["treatment", "count"]
        pp = g["sum"].sum() / g["count"].sum()
        z = (pt - pc) / np.sqrt(pp * (1 - pp) * (1 / g.loc["control", "count"] + 1 / g.loc["treatment", "count"]))
        rows.append({"day": d + 1, "users": int(g["count"].sum()), "rel_lift": pt / pc - 1, "z": z,
                     "p_value": 2 * (1 - norm.cdf(abs(z)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trap 2: novelty effect
# ---------------------------------------------------------------------------
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 11), (12, 16), (17, 27)]


def lift_by_exposure_age(sessions: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Treatment effect as a function of days since the user first saw the feature.
    This is the right x-axis for novelty: calendar day mixes fresh and seasoned users."""
    rows = []
    s = sessions.assign(treat=(sessions.variant == "treatment").astype(int))
    groups = [("all", s)] if by is None else list(s.groupby(by))
    for level, g in groups:
        for lo, hi in BINS:
            b = g[(g.days_since_exposure >= lo) & (g.days_since_exposure <= hi)]
            m = smf.ols("ordered ~ treat", data=b).fit(cov_type="cluster", cov_kwds={"groups": b["user_id"]})
            base = b.loc[b.treat == 0, "ordered"].mean()
            rows.append({"level": level, "bin": f"{lo}" if lo == hi else f"{lo}-{hi}", "mid": (lo + hi) / 2,
                         "sessions": len(b), "rel_lift": m.params["treat"] / base,
                         "ci_low": (m.params["treat"] - 1.96 * m.bse["treat"]) / base,
                         "ci_high": (m.params["treat"] + 1.96 * m.bse["treat"]) / base,
                         "weight": 1 / (m.bse["treat"] / base) ** 2})
    return pd.DataFrame(rows)


def fit_decay(curve: pd.DataFrame) -> dict:
    """Fit lift(k) = steady + bump * exp(-k / tau). 'steady' is the number that matters for the business."""
    f = lambda k, steady, bump, tau: steady + bump * np.exp(-k / tau)
    (steady, bump, tau), cov = curve_fit(f, curve["mid"], curve["rel_lift"], p0=[0.0, 0.1, 3.0],
                                         sigma=1 / np.sqrt(curve["weight"]), absolute_sigma=True,
                                         bounds=([-1, -1, 0.3], [1, 1, 30]), maxfev=20000)
    return {"steady_state": steady, "novelty_bump": bump, "tau_days": tau, "steady_se": float(np.sqrt(cov[0, 0]))}


def calendar_lift(sessions: pd.DataFrame) -> pd.DataFrame:
    g = sessions.groupby(["day", "variant"]).ordered.mean().unstack()
    return pd.DataFrame({"day": g.index + 1, "rel_lift": g.treatment / g.control - 1})
