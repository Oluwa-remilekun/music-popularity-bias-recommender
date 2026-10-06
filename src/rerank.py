"""Popularity-bias mitigation via post-processing re-ranking."""

import numpy as np


def _norm(x):
    """Min-max normalize to [0,1] within this candidate list."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rerank_popularity_penalty(candidate_ids, rel_scores, popularity, lam, N=10):
    """Blend relevance and un-popularity with dial lam, then take top-N.
    lam=0 reproduces the baseline order; lam=1 ignores relevance entirely."""
    candidate_ids = np.asarray(candidate_ids)
    rel = _norm(rel_scores)
    pop = _norm(popularity[candidate_ids])
    blended = (1 - lam) * rel - lam * pop
    order = np.argsort(-blended)[:N]
    return candidate_ids[order]
