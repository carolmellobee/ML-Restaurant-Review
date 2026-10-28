"""Topic modeling with NMF on TF-IDF.

Why NMF and not LDA: reviews are short (1-7 sentences). LDA's document-mixture
assumption needs more words per document to be stable; NMF on TF-IDF gives
tighter, more readable topics on short text and is faster to iterate. BERTopic
would be the next step on real data (see README), but it adds a heavy
dependency and a lot of knobs for a portfolio project.

k is chosen with an NPMI coherence score over a small grid, then the topics are
labeled by a human (us) by reading the top words. That last step is deliberate:
automatic topic labels are where most "topic modeling" projects quietly go wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Generic evaluative / filler words. They carry sentiment but no *aspect*, so
# left in they become junk topics like "highly recommend" or "never again".
# Sentiment is measured separately; topics should answer "about what?".
EXTRA_STOP = {
    "just", "really", "like", "got", "place", "time", "order", "ordered", "will",
    "did", "didn", "don", "ve", "ll", "im", "honestly", "pretty", "came", "new",
    "favorite", "second", "month", "definitely", "10", "ok", "plus", "also",
    "recommend", "recommended", "highly", "loved", "love", "liked", "amazing", "perfect",
    "good", "great", "best", "terrible", "awful", "disappointed", "disappointing",
    "never", "avoid", "waste", "decent", "special", "average", "meh", "expected",
    "probably", "won", "chance", "better", "deleting", "app", "refund", "asked",
    "experience", "solid", "choice", "feelings", "mixed", "come", "coming", "said",
    "wow", "nothing", "say", "favourite", "lot", "way", "bit", "food",
}


def _vectorizer() -> TfidfVectorizer:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    return TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.4, sublinear_tf=True)


def npmi_coherence(top_words: list[list[str]], texts: pd.Series) -> float:
    """Mean NPMI over pairs of top words, using document co-occurrence."""
    vocab = sorted({w for ws in top_words for w in ws})
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), binary=True)
    X = cv.fit_transform(texts).tocsc()
    n = X.shape[0]
    idx = {w: i for i, w in enumerate(vocab)}
    df = np.asarray(X.sum(0)).ravel()
    scores = []
    for ws in top_words:
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = idx[ws[i]], idx[ws[j]]
                co = X[:, a].multiply(X[:, b]).sum()
                if co == 0 or df[a] == 0 or df[b] == 0:
                    scores.append(-1.0)
                    continue
                p_ab, p_a, p_b = co / n, df[a] / n, df[b] / n
                pmi = np.log(p_ab / (p_a * p_b))
                scores.append(pmi / -np.log(p_ab) if p_ab < 1 else 1.0)
    return float(np.mean(scores))


def split_sentences(df: pd.DataFrame, text_col: str = "text", min_words: int = 3) -> pd.DataFrame:
    """One row per sentence. Reviews mix aspects ("great pasta, late driver"), so
    topics are far cleaner at sentence level than at review level."""
    s = df.assign(sentence=df[text_col].str.split(r"(?<=[.!?])\s+")).explode("sentence")
    s["sentence"] = s["sentence"].str.strip()
    return s[s["sentence"].str.split().str.len() >= min_words].reset_index(drop=True)


def fit_nmf(texts: pd.Series, k: int, seed: int = 42):
    vec = _vectorizer()
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=1000)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    top = [list(terms[np.argsort(-row)[:8]]) for row in nmf.components_]
    return W, top


def unigram_top(top: list[list[str]], n: int = 6) -> list[list[str]]:
    """Coherence is computed on unigrams only; bigrams trivially co-occur with
    their own parts and would inflate the score."""
    return [[w for w in ws if " " not in w][:n] for ws in top]


def choose_k(texts: pd.Series, ks=range(5, 15), seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in ks:
        _, top = fit_nmf(texts, k, seed)
        rows.append({"k": k, "npmi": npmi_coherence(unigram_top(top), texts)})
    return pd.DataFrame(rows)


def assign_topics(W: np.ndarray, labels: list[str], min_weight: float = 0.02) -> pd.DataFrame:
    """Dominant topic per review, plus a 'none' bucket for reviews no topic explains."""
    dom = W.argmax(1)
    strength = W.max(1)
    lab = np.where(strength < min_weight, "no_clear_topic", np.array(labels)[dom])
    return pd.DataFrame({"topic": lab, "topic_weight": strength})


def topic_rating_table(df: pd.DataFrame, W: np.ndarray, labels: list[str], threshold: float = 0.05) -> pd.DataFrame:
    """For each topic: how many reviews touch it, and the mean rating of those reviews."""
    rows = []
    for j, lab in enumerate(labels):
        mask = W[:, j] > threshold
        rows.append({
            "topic": lab,
            "share_of_reviews": mask.mean(),
            "mean_rating": df.loc[mask, "rating"].mean(),
            "share_1_2_star": (df.loc[mask, "rating"] <= 2).mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_rating")


def suggest_labels(sentences: pd.Series, W: np.ndarray, tagger, categories: list[str], top_n: int = 60) -> pd.DataFrame:
    """Suggest a label per topic: which taxonomy category dominates its most typical sentences.

    'purity' = share of those sentences carrying that category. Low purity means the
    topic is mixing things (or is filler), which is the cue for a human to look.
    This is a reading aid, not ground truth: a human still signs off on the labels.
    """
    rows = []
    for j in range(W.shape[1]):
        idx = np.argsort(-W[:, j])[:top_n]
        tags = tagger(sentences.iloc[idx].reset_index(drop=True))[categories]
        share = tags.mean().sort_values(ascending=False)
        second = share.index[1] if share.iloc[1] >= 0.25 else ""
        rows.append({"topic": j, "suggested_label": share.index[0] if share.iloc[0] >= 0.5 else "mixed_or_filler",
                     "purity": share.iloc[0], "also_contains": second})
    return pd.DataFrame(rows)
