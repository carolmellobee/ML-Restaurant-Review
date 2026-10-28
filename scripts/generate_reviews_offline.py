"""Offline fallback: write review text from phrase pools, no API key needed.

This exists so the repo runs end to end for anyone who clones it. It follows the
exact same latent structure as the LLM generator (src/generation.py), so every
downstream notebook works with either source.

Be honest about what this is: template text is easier for NLP than real or LLM
text (less vocabulary drift, no ambiguity beyond what we planted). Treat the
pipeline's accuracy on this data as an upper bound. Rerun with
scripts/generate_reviews_llm.py for a harder, more realistic corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REVIEWS_CSV, SEED  # noqa: E402
from src.generation import sample_review_skeletons  # noqa: E402

DISHES = {
    "Italian": ["lasagna", "carbonara", "risotto", "tiramisu"],
    "Japanese": ["ramen", "sushi combo", "gyoza", "temaki"],
    "Brazilian": ["feijoada", "picanha", "coxinha", "pão de queijo"],
    "Mexican": ["tacos", "burrito", "nachos", "quesadilla"],
    "Burgers": ["smash burger", "chicken sandwich", "milkshake", "bacon burger"],
    "Indian": ["butter chicken", "biryani", "naan", "samosas"],
    "Vegan": ["buddha bowl", "vegan burger", "falafel wrap", "acai bowl"],
    "Pizza": ["margherita", "pepperoni pizza", "calzone", "garlic knots"],
}

PHRASES = {
    "delivery_delay": [
        "the delivery took over an hour and a half",
        "it showed 30 minutes and arrived after 80",
        "the driver was late, like really late",
        "we waited forever for the delivery",
        "the ETA kept going up and the order showed up super late",
        "late delivery again, second time this month",
        "took ages to arrive, we almost cancelled",
    ],
    "cold_food": [
        "the food arrived cold",
        "the {dish} was lukewarm and soggy",
        "everything was cold by the time it got here",
        "fries were soggy and cold",
        "had to microwave the {dish} because it was cold",
    ],
    "order_accuracy": [
        "they forgot half of the order",
        "the {dish} was missing from the bag",
        "got someone else's order",
        "wrong items, I ordered the {dish} and got something else",
        "missing items and no napkins or cutlery",
    ],
    "rude_staff": [
        "the staff was rude when I asked a question",
        "the driver was rude and threw the bag at me",
        "the manager was dismissive and rude",
        "the waiter rolled his eyes at us",
        "very unfriendly attitude from the staff",
    ],
    "price_value": [
        "way too expensive for what you get",
        "overpriced, honestly",
        "the price went up but the quality did not",
        "not worth the money",
        "pricey for such a basic {dish}",
    ],
    "small_portion": [
        "the portion was tiny",
        "small portions for the price",
        "the {dish} portion was barely enough for one person",
        "they skimp on the portions now",
    ],
    "hygiene": [
        "found a hair in my {dish}",
        "the place was dirty and the bathroom was disgusting",
        "I got sick after eating here",
        "the container was greasy and dirty outside",
        "food poisoning, never again",
    ],
    "long_wait_dinein": [
        "we waited 40 minutes for a table even with a reservation",
        "service was painfully slow",
        "the food took forever to come out of the kitchen",
        "we waited ages just to get the check",
    ],
    "taste": [
        "the {dish} was delicious",
        "amazing flavor",
        "the {dish} tasted incredible",
        "best {dish} in the neighborhood",
        "really tasty, well seasoned",
        "the food was delicious",
    ],
    "friendly_staff": [
        "the staff was super friendly",
        "lovely team, very welcoming",
        "the waiter was kind and attentive",
        "friendly driver, very polite",
    ],
    "fast_delivery": [
        "delivery was fast",
        "arrived earlier than expected",
        "super quick delivery",
        "it got here in 20 minutes",
    ],
    "good_value": [
        "great value for the price",
        "fair price",
        "cheap and good",
        "worth every penny",
    ],
    "generous_portion": [
        "huge portions",
        "the portion was generous, had leftovers",
        "plenty of food",
    ],
    "ambience": [
        "cozy atmosphere",
        "nice music and decor",
        "great vibe for a date night",
    ],
}

OPENERS = {
    1: ["Terrible experience.", "Never again.", "Really disappointed.", "Awful.", "What a waste.", ""],
    2: ["Not great.", "Pretty disappointing.", "Expected more.", "Meh.", ""],
    3: ["It was ok.", "Mixed feelings.", "Decent but not special.", "Average.", ""],
    4: ["Pretty good!", "Good experience.", "Solid choice.", "Liked it.", ""],
    5: ["Loved it!", "Amazing!", "Perfect.", "10/10.", "So good.", ""],
}

CLOSERS = {
    1: ["Avoid.", "Asked for a refund.", "Deleting the app.", "Do better.", ""],
    2: ["Probably won't order again.", "Might give it another chance.", ""],
    3: ["Might come back.", "Could be better.", ""],
    4: ["Will order again.", "Recommended.", ""],
    5: ["Will definitely be back!", "Highly recommend!", "My new favorite.", ""],
}

FILLERS = [
    "Ordered on a Friday night.", "We were a group of four.", "First time trying this place.",
    "I've been here a few times.", "Came after work.", "It was raining.", "Used a coupon.",
]

SARCASM = [
    "Great, just great.", "Loved waiting, said no one ever.", "Wow, what service.",
]


def _typo(rng, text: str) -> str:
    if rng.random() > 0.15 or len(text) < 20:
        return text
    i = rng.integers(1, len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


DINE_IN_WORDS = ("waiter", "table", "bathroom", "place was", "kitchen", "check", "decor", "atmosphere", "vibe")
DELIVERY_WORDS = ("driver", "bag", "deliver", "arrive", "eta", "got here", "microwave", "container")


def _wrong_channel(phrase: str, channel: str) -> bool:
    p = phrase.lower()
    words = DELIVERY_WORDS if channel == "dine_in" else DINE_IN_WORDS
    return any(w in p for w in words)


def write_review(row, rng: np.random.Generator) -> str:
    dishes = DISHES[row.cuisine]
    parts = []
    opener = rng.choice(OPENERS[row.rating])
    if opener:
        parts.append(opener)
    if rng.random() < 0.3:
        parts.append(rng.choice(FILLERS))

    aspects = [a for a in (row.true_praises + "|" + row.true_complaints).split("|") if a]
    rng.shuffle(aspects)
    for a in aspects:
        pool = [p for p in PHRASES[a] if not _wrong_channel(p, row.channel)] or PHRASES[a]
        phrase = rng.choice(pool).format(dish=rng.choice(dishes))
        if rng.random() < 0.5 and parts:
            phrase = rng.choice(["and ", "also ", "plus ", "but ", ""]) + phrase
        parts.append(phrase[0].upper() + phrase[1:] + rng.choice([".", ".", "!", "..."]))

    if not aspects:
        parts.append(rng.choice([f"Had the {rng.choice(dishes)}.", "Nothing much to say.", "It was food."]))
    if row.rating <= 2 and rng.random() < 0.12:
        parts.append(rng.choice(SARCASM))
    closer = rng.choice(CLOSERS[row.rating])
    if closer:
        parts.append(closer)

    text = " ".join(parts)
    if row.style == "casual with slang" and rng.random() < 0.5:
        text = text.lower()
    return _typo(rng, text)


def main() -> None:
    rng = np.random.default_rng(SEED + 1)
    skel = sample_review_skeletons()
    skel["text"] = [write_review(r, rng) for r in skel.itertuples()]
    skel["model"] = "offline-templates"
    out = skel.drop(columns=["persona", "style", "length"])
    REVIEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEWS_CSV, index=False)
    print(f"wrote {len(out)} reviews -> {REVIEWS_CSV}")


if __name__ == "__main__":
    main()
