"""Generate review TEXT with an LLM (Claude), following pre-sampled latent structure.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_reviews_llm.py                # all 3,000 reviews
    python scripts/generate_reviews_llm.py --limit 50     # quick smoke test

Cost note: with a small model (default claude-haiku-5-5) 3,000 short reviews is
roughly 3,000 requests of ~250 input / ~120 output tokens. Check current pricing
before running the full set.

The script is resumable: finished reviews are cached to data/raw/_llm_cache.jsonl,
so a crash or rate-limit just means you run it again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, REVIEWS_CSV  # noqa: E402
from src.generation import sample_review_skeletons  # noqa: E402

CACHE = DATA_RAW / "_llm_cache.jsonl"
MODEL = os.environ.get("REVIEW_MODEL", "claude-haiku-5-5")

HUMAN_LABELS = {
    "delivery_delay": "the delivery arrived much later than promised",
    "cold_food": "the food arrived cold or soggy",
    "order_accuracy": "items were missing or the order was wrong",
    "rude_staff": "the staff or driver was rude",
    "price_value": "it felt overpriced for what you got",
    "small_portion": "portions were small",
    "hygiene": "a hygiene problem (dirty, hair in food, or felt sick after)",
    "long_wait_dinein": "a long wait for a table or for the food at the restaurant",
    "taste": "the food tasted great",
    "friendly_staff": "the staff was friendly",
    "fast_delivery": "delivery was fast",
    "good_value": "good value for money",
    "generous_portion": "generous portions",
    "ambience": "nice atmosphere in the restaurant",
}

SYSTEM = (
    "You write realistic customer reviews for a restaurant review app. "
    "Real reviews are uneven: people ramble, skip punctuation, exaggerate, mention one "
    "specific dish, and do NOT neatly list every point. Never use the words 'overall' or "
    "'in conclusion'. Never mention star numbers. Output only the review text."
)


def build_prompt(row: pd.Series) -> str:
    def fmt(col):
        items = [HUMAN_LABELS[x] for x in str(row[col]).split("|") if x]
        return "; ".join(items) if items else "nothing in particular"

    return (
        f"Restaurant: {row.restaurant_name} ({row.cuisine}, price {row.price_tier}).\n"
        f"Experience: {'ordered delivery' if row.channel == 'delivery' else 'ate at the restaurant'}.\n"
        f"Reviewer: {row.persona}. Writing style: {row['style']}. Length: {row['length']}.\n"
        f"They would rate it {row.rating} out of 5 (do not state the number).\n"
        f"Things that went wrong (mention each, in your own words): {fmt('true_complaints')}.\n"
        f"Things that went well (mention each, in your own words): {fmt('true_praises')}.\n"
        "Do not invent other problems."
    )


def generate_one(client, row: pd.Series) -> dict:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=1.0,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(row)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return {"review_id": row.review_id, "text": text, "model": MODEL}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (or use scripts/generate_reviews_offline.py).")

    import anthropic

    client = anthropic.Anthropic(max_retries=6)
    skel = sample_review_skeletons()
    if args.limit:
        skel = skel.head(args.limit)

    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            done[rec["review_id"]] = rec
    todo = skel[~skel.review_id.isin(done)]
    print(f"{len(done)} cached, {len(todo)} to generate with {MODEL}")

    with CACHE.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(generate_one, client, row): row.review_id for _, row in todo.iterrows()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # keep going; rerun picks up failures
                print(f"failed {futures[fut]}: {exc}")
                continue
            done[rec["review_id"]] = rec
            fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")

    texts = pd.DataFrame(done.values())[["review_id", "text", "model"]]
    out = skel.merge(texts, on="review_id", how="inner")
    out = out.drop(columns=["persona", "style", "length"])
    out.to_csv(REVIEWS_CSV, index=False)
    print(f"wrote {len(out)} reviews -> {REVIEWS_CSV}")


if __name__ == "__main__":
    main()
