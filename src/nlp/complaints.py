"""Complaint categories: a transparent lexicon tagger, validated against ground truth,
then tied to star ratings.

Why a lexicon and not only topics: the topic model (topics.py) finds the big
themes but misses rare-and-severe ones (hygiene shows up in ~2% of reviews and
never becomes its own topic). A business taxonomy needs every category the ops
team acts on, with a measurable precision/recall. A lexicon is the auditable
baseline; a zero-shot LLM classifier is the upgrade (see README).

Caveat we state openly: the lexicon was written after reading a sample of the
corpus, so its scores on this synthetic data are optimistic. On real data,
hand-label ~300 reviews and score against those instead.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..generation import COMPLAINTS, PRAISES

LEXICON = {
    # --- complaints ---------------------------------------------------------
    "delivery_delay": r"\b(late|forever|took (ages|over)|hour and a half|eta|waited (forever|ages)|after \d+|almost cancel\w*)\b",
    "cold_food": r"\b(cold|lukewarm|soggy|microwave\w*)\b",
    "order_accuracy": r"\b(forgot|missing|wrong (item|order)\w*|someone else'?s order|got something else)\b",
    "rude_staff": r"\b(rude|dismissive|rolled (his|her) eyes|unfriendly|threw)\b",
    "price_value": r"\b(expensive|overpriced|pricey|not worth|price went up)\b",
    "small_portion": r"\b(tiny|small portions?|barely enough|skimp\w*)\b",
    "hygiene": r"\b(hair|dirty|disgusting|got sick|food poisoning|greasy)\b",
    "long_wait_dinein": r"\b(for a table|painfully slow|out of the kitchen|get the check|slow service)\b",
    # --- praises (used by the Review Highlights module) -------------------------
    "taste": r"\b(delicious|tasty|tasted|flavou?r|seasoned|best \w+ in)\b",
    "friendly_staff": r"(?<!un)\b(friendly|kind and|attentive|welcoming|polite|lovely team)\b",
    "fast_delivery": r"(?<!not )\b(fast|quick|earlier than expected|in \d+ minutes)\b",
    "good_value": r"\b(great value|fair price|cheap and good|worth every penny)\b",
    "generous_portion": r"\b(huge portions?|generous|plenty of food|leftovers)\b",
    "ambience": r"\b(cozy|atmosphere|decor|vibe)\b",
}


def tag(texts: pd.Series) -> pd.DataFrame:
    out = {}
    lower = texts.str.lower()
    for cat, pat in LEXICON.items():
        pat = re.sub(r"\((?!\?)", "(?:", pat)   # non-capturing groups: we only need a match/no-match
        out[cat] = lower.str.contains(pat, regex=True)
    return pd.DataFrame(out, index=texts.index).astype(int)


def evaluate(df: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 of each category against the latent ground truth."""
    rows = []
    for cat in COMPLAINTS + PRAISES:
        truth_col = "true_complaints" if cat in COMPLAINTS else "true_praises"
        truth = df[truth_col].fillna("").str.split("|").apply(lambda xs: cat in xs).astype(int)
        pred = tags[cat]
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        rows.append({
            "category": cat, "kind": "complaint" if cat in COMPLAINTS else "praise",
            "support": int(truth.sum()), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p and r else np.nan,
        })
    return pd.DataFrame(rows)


def prevalence_by_rating(df: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews at each star level that mention each complaint."""
    return tags[COMPLAINTS].groupby(df["rating"]).mean().T


def star_impact(df: pd.DataFrame, tags: pd.DataFrame):
    """OLS: rating ~ complaint flags + controls.

    Coefficient = average star penalty associated with a complaint, holding the
    other complaints, channel and price tier fixed. This is association, not
    causation (people who are angry for other reasons may also notice cold food),
    which is exactly why we follow up with an experiment.
    HC3 robust SEs because rating residuals are heteroskedastic (bounded 1-5).
    """
    data = pd.concat([df[["rating", "channel", "price_tier", "restaurant_id"]], tags[COMPLAINTS]], axis=1)
    formula = "rating ~ " + " + ".join(COMPLAINTS) + " + C(channel) + C(price_tier)"
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    coefs = model.params[COMPLAINTS].rename("star_impact").to_frame()
    coefs["ci_low"] = model.conf_int().loc[COMPLAINTS, 0]
    coefs["ci_high"] = model.conf_int().loc[COMPLAINTS, 1]
    return coefs, model


def priority_matrix(df: pd.DataFrame, tags: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    """Frequency x severity. 'Stars lost per 100 reviews' is the number to rank by:
    a severe-but-rare problem can matter less than a mild-but-everywhere one."""
    pm = impact[["star_impact"]].copy()
    pm["prevalence"] = tags[COMPLAINTS].mean()
    pm["share_of_1_2_star_reviews"] = tags.loc[df.rating <= 2, COMPLAINTS].mean()
    pm["stars_lost_per_100_reviews"] = -pm["star_impact"] * pm["prevalence"] * 100
    return pm.sort_values("stars_lost_per_100_reviews", ascending=False)
