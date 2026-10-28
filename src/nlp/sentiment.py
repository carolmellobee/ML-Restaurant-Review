"""Sentiment: a lexicon baseline (VADER) and a supervised model, compared to stars.

Stars are already a sentiment label, so why bother? Two reasons that come up in
interviews:
  1. Text sentiment catches the *mixed* 3-4 star reviews ("food great, delivery
     awful") that a star average flattens.
  2. The gap between text and stars is informative: sarcasm, polite-but-angry
     reviewers, and rating inflation all show up there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def star_bucket(rating: pd.Series) -> pd.Series:
    return pd.cut(rating, bins=[0, 2, 3, 5], labels=["negative", "neutral", "positive"]).astype(str)


def add_vader(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    out = df.copy()
    out["vader_compound"] = [sia.polarity_scores(t)["compound"] for t in out[text_col]]
    out["vader_label"] = pd.cut(
        out["vader_compound"], bins=[-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"]
    ).astype(str)
    return out


def supervised_sentiment(df: pd.DataFrame, text_col: str = "text", seed: int = 42):
    """5-fold out-of-fold predictions of the star bucket from text (TF-IDF + LR)."""
    y = star_bucket(df["rating"])
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, df[text_col], y, cv=cv)
    return pred, classification_report(y, pred, output_dict=True, zero_division=0)


def compare_to_stars(df: pd.DataFrame) -> dict:
    y = star_bucket(df["rating"])
    rho, p = spearmanr(df["vader_compound"], df["rating"])
    return {
        "spearman_vader_vs_stars": rho,
        "spearman_p": p,
        "vader_macro_f1": f1_score(y, df["vader_label"], average="macro"),
        "supervised_macro_f1": f1_score(y, df["supervised_label"], average="macro"),
    }


def disagreements(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Reviews where text sentiment and stars point opposite ways."""
    bad = df[((df.rating <= 2) & (df.vader_compound > 0.4)) | ((df.rating >= 4) & (df.vader_compound < -0.4))]
    return bad.sort_values("vader_compound")[["rating", "vader_compound", "text"]].head(n)


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = add_vader(df)
    pred, _ = supervised_sentiment(df)
    df["supervised_label"] = pred
    return df, compare_to_stars(df)


if __name__ == "__main__":  # pragma: no cover
    from src.config import REVIEWS_CSV

    d, m = run(pd.read_csv(REVIEWS_CSV))
    print({k: round(v, 3) for k, v in m.items()})
    print(disagreements(d).to_string())
    print(np.round(d.groupby("rating").vader_compound.mean(), 3))
