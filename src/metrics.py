"""Evaluation metrics: accuracy + popularity-bias.

Each metric is validated on a tiny hand-computed example before use at scale,
because a wrong metric does not crash -- it silently returns plausible but
incorrect numbers.
"""

import math
import numpy as np


# ---------- accuracy (higher = better) ----------

def recall_at_k(recommended_ids, holdout_ids, k=10):
    """Fraction of a user's held-out artists that appear in the top-k recommendations."""
    if len(holdout_ids) == 0:
        return 0.0
    hits = sum(1 for i in recommended_ids[:k] if i in holdout_ids)
    return hits / len(holdout_ids)


def ndcg_at_k(recommended_ids, holdout_ids, k=10):
    """NDCG@k: like recall but rewards ranking correct artists nearer the top."""
    dcg = 0.0
    for rank, i in enumerate(recommended_ids[:k], start=1):
        if i in holdout_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(holdout_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------- popularity bias (lower ARP / Gini = less biased; higher coverage = better) ----------

def average_recommendation_popularity(recs, popularity):
    """Mean popularity of all recommended artists across users. Lower = less biased.
    recs: dict {user: [artist_ids]}; popularity: array indexed by artist id."""
    vals = [popularity[i] for items in recs.values() for i in items]
    return float(np.mean(vals)) if vals else 0.0


def catalog_coverage(recs, n_artists):
    """Fraction of the whole catalog that appears in ANY user's recommendations."""
    seen = set()
    for items in recs.values():
        seen.update(items)
    return len(seen) / n_artists


def gini_index(recs, n_artists):
    """Gini of recommendation exposure across artists. 0 = perfectly even, higher = concentrated."""
    exposure = np.zeros(n_artists)
    for items in recs.values():
        for i in items:
            exposure[i] += 1
    c = np.sort(exposure[exposure > 0])
    n = len(c)
    if n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * c) / (n * np.sum(c))) - (n + 1) / n)
