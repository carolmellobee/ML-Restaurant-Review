"""The analysis itself: validity checks first, then the primary metric, guardrails,
variance reduction, and pre-registered segments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, chisquare, norm, ttest_ind


# ---------------------------------------------------------------------------
# 0. Sample ratio mismatch: if the split isn't 50/50, stop. Nothing else is trustworthy.
# ---------------------------------------------------------------------------
def srm_check(users: pd.DataFrame, expected=(0.5, 0.5)) -> dict:
    counts = users["variant"].value_counts().reindex(["control", "treatment"])
    exp = np.array(expected) * counts.sum()
    stat, p = chisquare(counts.to_numpy(), exp)
    return {"control": int(counts["control"]), "treatment": int(counts["treatment"]), "chi2": stat, "p_value": p,
            "srm_detected": p < 0.001}  # strict threshold: SRM checks run on every test, so keep false alarms rare


# ---------------------------------------------------------------------------
# 1. Primary metric: two-proportion comparison with absolute and relative CIs
# ---------------------------------------------------------------------------
def compare_proportions(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> dict:
    g = df.groupby("variant")[metric].agg(["sum", "count"])
    xc, nc = g.loc["control"]
    xt, nt = g.loc["treatment"]
    pc, pt = xc / nc, xt / nt
    se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    z_crit = norm.ppf(1 - alpha / 2)
    diff = pt - pc
    # test statistic with pooled variance under H0
    pp = (xc + xt) / (nc + nt)
    z = diff / np.sqrt(pp * (1 - pp) * (1 / nc + 1 / nt))
    # relative lift CI via delta method on log(pt/pc)
    rel = pt / pc - 1
    se_log = np.sqrt((1 - pc) / (pc * nc) + (1 - pt) / (pt * nt))
    return {
        "n_control": int(nc), "n_treatment": int(nt),
        "rate_control": pc, "rate_treatment": pt,
        "abs_diff": diff, "abs_ci": (diff - z_crit * se, diff + z_crit * se),
        "rel_lift": rel,
        "rel_ci": (np.exp(np.log(pt / pc) - z_crit * se_log) - 1, np.exp(np.log(pt / pc) + z_crit * se_log) - 1),
        "z": z, "p_value": 2 * (1 - norm.cdf(abs(z))),
    }


# ---------------------------------------------------------------------------
# 2. CUPED: use a pre-period covariate to shrink variance (same estimand, tighter CI)
# ---------------------------------------------------------------------------
def cuped(df: pd.DataFrame, metric: str, covariate: str = "pre_period_orders") -> dict:
    y, x = df[metric].astype(float), df[covariate].astype(float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    adj = y - theta * (x - x.mean())
    out = {}
    for name, series in [("raw", y), ("cuped", adj)]:
        t = series[df.variant == "treatment"]
        c = series[df.variant == "control"]
        diff = t.mean() - c.mean()
        se = np.sqrt(t.var() / len(t) + c.var() / len(c))
        out[name] = {"diff": diff, "se": se, "ci": (diff - 1.96 * se, diff + 1.96 * se)}
    out["theta"] = theta
    out["variance_reduction"] = 1 - (out["cuped"]["se"] / out["raw"]["se"]) ** 2
    return out


# ---------------------------------------------------------------------------
# 3. Guardrails: things that must not get worse
# ---------------------------------------------------------------------------
def guardrails(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conv = users[users.converted_7d == 1]
    t, c = conv[conv.variant == "treatment"].orders_7d, conv[conv.variant == "control"].orders_7d
    rows.append({"guardrail": "orders per converter (7d)", "control": c.mean(), "treatment": t.mean(),
                 "rel_change": t.mean() / c.mean() - 1, "p_value": ttest_ind(t, c, equal_var=False).pvalue})
    o = users.groupby("variant")[["refunds_7d", "orders_7d"]].sum()
    rc = o.loc["control", "refunds_7d"] / o.loc["control", "orders_7d"]
    rt = o.loc["treatment", "refunds_7d"] / o.loc["treatment", "orders_7d"]
    table = np.array([[o.loc[v, "refunds_7d"], o.loc[v, "orders_7d"] - o.loc[v, "refunds_7d"]] for v in ["control", "treatment"]])
    rows.append({"guardrail": "refund rate (per order)", "control": rc, "treatment": rt,
                 "rel_change": rt / rc - 1, "p_value": chi2_contingency(table)[1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Pre-registered segments, with a multiple-comparison correction
# ---------------------------------------------------------------------------
def segment_effects(users: pd.DataFrame, segments: list[str], metric: str = "converted_7d") -> pd.DataFrame:
    rows = []
    for seg in segments:
        for level, sub in users.groupby(seg):
            r = compare_proportions(sub, metric)
            rows.append({"segment": seg, "level": level, "n": r["n_control"] + r["n_treatment"],
                         "rate_control": r["rate_control"], "rel_lift": r["rel_lift"],
                         "rel_ci_low": r["rel_ci"][0], "rel_ci_high": r["rel_ci"][1], "p_value": r["p_value"]})
    out = pd.DataFrame(rows)
    m = len(out)
    out["p_bonferroni"] = np.minimum(out["p_value"] * m, 1.0)
    return out


def interaction_test(users: pd.DataFrame, segment: str, metric: str = "converted_7d"):
    """Is the treatment effect *different* across levels? (Not the same as 'significant in one, not the other'.)
    Linear probability model with HC1 SEs: coefficients read directly in percentage points."""
    d = users.assign(treat=(users.variant == "treatment").astype(int))
    return smf.ols(f"{metric} ~ treat * C({segment})", data=d).fit(cov_type="HC1")


# ---------------------------------------------------------------------------
# 5. Long-run (post-novelty) effect at session level, clustered by user
# ---------------------------------------------------------------------------
def long_run_effect(sessions: pd.DataFrame, min_days_since: int = 7, by: str | None = None) -> pd.DataFrame:
    """Sessions are not independent (one user has many), so naive SEs are too small.
    Cluster-robust SEs by user fix the unit-of-analysis mismatch."""
    s = sessions[sessions.days_since_exposure >= min_days_since].assign(
        treat=lambda d: (d.variant == "treatment").astype(int))
    groups = [("all", s)] if by is None else list(s.groupby(by))
    rows = []
    for level, g in groups:
        m = smf.ols("ordered ~ treat", data=g).fit(cov_type="cluster", cov_kwds={"groups": g["user_id"]})
        naive = smf.ols("ordered ~ treat", data=g).fit()
        base = g.loc[g.treat == 0, "ordered"].mean()
        b, se = m.params["treat"], m.bse["treat"]
        rows.append({"level": level, "sessions": len(g), "cvr_control": base, "rel_lift": b / base,
                     "rel_ci_low": (b - 1.96 * se) / base, "rel_ci_high": (b + 1.96 * se) / base,
                     "p_value": m.pvalues["treat"], "se_cluster": se, "se_naive": naive.bse["treat"]})
    return pd.DataFrame(rows)
