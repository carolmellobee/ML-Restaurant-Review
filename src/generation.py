"""Latent 'ground truth' for each review, shared by the LLM and offline generators.

Why this exists
---------------
If you just ask an LLM for "3,000 restaurant reviews", you get reviews that are
too polite, too uniform, and rated mostly 4-5 stars. Worse, you have no way to
check whether your NLP pipeline actually works, because you don't know what was
in the data.

So we sample the *structure* first (rating, channel, which complaints occurred,
how severe they were) from a distribution that mirrors real review platforms
(J-shaped ratings, complaints concentrated in 1-2 stars), and only then ask the
LLM to *write* the text. The NLP notebooks never see these latent columns; we
only use them at the end to score how well the pipeline recovered the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CUISINES, N_REVIEWS, SEED

COMPLAINTS = [
    "delivery_delay",   # late delivery / long ETA miss
    "cold_food",        # arrived cold or soggy
    "order_accuracy",   # missing or wrong items
    "rude_staff",       # staff / driver attitude
    "price_value",      # too expensive for what you get
    "small_portion",    # portion size
    "hygiene",          # hair in food, dirty, food poisoning
    "long_wait_dinein", # waited for a table / slow service in restaurant
]

PRAISES = ["taste", "friendly_staff", "fast_delivery", "good_value", "generous_portion", "ambience"]

CONTRADICTS = {
    "fast_delivery": ("delivery_delay",),
    "good_value": ("price_value",),
    "generous_portion": ("small_portion",),
    "friendly_staff": ("rude_staff",),
}

# Real platforms are J-shaped: lots of 5s, a bump of angry 1s, few 2-3s.
RATING_PROBS = {1: 0.13, 2: 0.08, 3: 0.11, 4: 0.24, 5: 0.44}

# Probability a complaint is present, by star rating. Columns follow COMPLAINTS.
# Planted signal the pipeline should recover:
#   * delivery_delay is the most common complaint in 1-2 star reviews
#   * hygiene is rare but almost always means 1 star
#   * price_value is common even in 3-4 star reviews (mild complaint)
COMPLAINT_PROBS = {
    #        delay cold  acc   rude  price small hyg   wait
    1: [0.55, 0.35, 0.30, 0.25, 0.15, 0.10, 0.12, 0.10],
    2: [0.45, 0.30, 0.25, 0.15, 0.25, 0.15, 0.03, 0.12],
    3: [0.25, 0.15, 0.12, 0.06, 0.35, 0.22, 0.005, 0.12],
    4: [0.08, 0.04, 0.04, 0.02, 0.20, 0.10, 0.0, 0.05],
    5: [0.01, 0.00, 0.01, 0.00, 0.03, 0.02, 0.0, 0.01],
}

PRAISE_PROBS = {
    #        taste friendly fast  value portion ambience
    1: [0.05, 0.02, 0.00, 0.01, 0.02, 0.02],
    2: [0.20, 0.05, 0.02, 0.03, 0.05, 0.05],
    3: [0.45, 0.15, 0.08, 0.10, 0.12, 0.12],
    4: [0.75, 0.30, 0.25, 0.25, 0.25, 0.20],
    5: [0.90, 0.45, 0.35, 0.30, 0.35, 0.30],
}

PERSONAS = [
    "a busy office worker ordering lunch",
    "a college student on a budget",
    "a parent ordering for a family of four",
    "a food enthusiast who notices details",
    "a retiree who rarely writes reviews",
    "someone celebrating a birthday",
    "a night-shift nurse ordering late",
    "a tourist visiting the city",
]

STYLES = ["short and blunt", "detailed and measured", "casual with slang", "sarcastic", "formal", "emotional"]


def _restaurant_names(rng: np.random.Generator, n: int) -> list[str]:
    first = ["Casa", "Bella", "Golden", "Little", "Urban", "Sabor", "Green", "Blue", "Old", "Nonna's"]
    second = ["Fork", "Kitchen", "Grill", "Table", "Spoon", "Garden", "House", "Corner", "Bistro", "Oven"]
    names = {f"{a} {b}" for a in first for b in second}
    return list(rng.choice(sorted(names), size=n, replace=False))


def sample_review_skeletons(n: int = N_REVIEWS, seed: int = SEED) -> pd.DataFrame:
    """Return one row per review with the latent structure the text must follow."""
    rng = np.random.default_rng(seed)

    n_restaurants = 60
    rest = pd.DataFrame({
        "restaurant_id": [f"R{i:03d}" for i in range(n_restaurants)],
        "restaurant_name": _restaurant_names(rng, n_restaurants),
        "cuisine": rng.choice(CUISINES, size=n_restaurants),
        "price_tier": rng.choice(["$", "$$", "$$$"], size=n_restaurants, p=[0.35, 0.45, 0.20]),
        # some restaurants are just worse: shifts their rating distribution
        "quality": rng.normal(0, 0.6, size=n_restaurants),
    })

    rows = []
    stars = np.array(list(RATING_PROBS))
    base_p = np.array(list(RATING_PROBS.values()))
    for i in range(n):
        r = rest.iloc[rng.integers(n_restaurants)]
        # tilt the J-curve by restaurant quality
        logits = np.log(base_p) + r.quality * (stars - 3) * 0.35
        p = np.exp(logits) / np.exp(logits).sum()
        rating = int(rng.choice(stars, p=p))
        channel = rng.choice(["delivery", "dine_in"], p=[0.7, 0.3])

        comp_p = np.array(COMPLAINT_PROBS[rating], dtype=float)
        if channel == "dine_in":
            comp_p[[0, 1]] = 0.0            # can't have a late delivery when dining in
            comp_p[7] *= 2.5
        else:
            comp_p[7] = 0.0
        if r.price_tier == "$$$":
            comp_p[4] = min(1.0, comp_p[4] * 1.6)
        complaints = [c for c, pc in zip(COMPLAINTS, comp_p) if rng.random() < pc]

        praise_p = np.array(PRAISE_PROBS[rating], dtype=float)
        if channel == "dine_in":
            praise_p[2] = 0.0
        else:
            praise_p[5] *= 0.2
        praises = [c for c, pc in zip(PRAISES, praise_p) if rng.random() < pc]
        # a review can't both praise and complain about the same thing
        praises = [p_ for p_ in praises if not any(c in complaints for c in CONTRADICTS.get(p_, ()))]

        rows.append({
            "review_id": f"V{i:05d}",
            "restaurant_id": r.restaurant_id,
            "restaurant_name": r.restaurant_name,
            "cuisine": r.cuisine,
            "price_tier": r.price_tier,
            "channel": channel,
            "rating": rating,
            "true_complaints": "|".join(complaints),
            "true_praises": "|".join(praises),
            "persona": rng.choice(PERSONAS),
            "style": rng.choice(STYLES),
            "length": rng.choice(["1-2 sentences", "3-4 sentences", "5-7 sentences"], p=[0.35, 0.45, 0.20]),
            "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 270)))).date(),
        })
    return pd.DataFrame(rows)
