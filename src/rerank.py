"""Popularity-bias mitigation via post-processing re-ranking."""

import numpy as np


def _norm(x):
    """Min-max normalize to [0,1] within this candidate list."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rerank_popularity_penalty(candidate_ids, rel_scores, popularity, lam, N=10):
    """Method 1. Blend relevance and un-popularity with dial lam, then take top-N."""
    candidate_ids = np.asarray(candidate_ids)
    rel = _norm(rel_scores)
    pop = _norm(popularity[candidate_ids])
    blended = (1 - lam) * rel - lam * pop
    order = np.argsort(-blended)[:N]
    return candidate_ids[order]


def rerank_binary_xquad(candidate_ids, rel_scores, is_head, p_tail, lam, N=10):
    """Method 2. xQuAD (long-tail promoting). Greedily builds the top-N; at each slot it rewards
    adding a long-tail artist, with the reward shrinking as the list's tail share approaches the
    user's own tail propensity p_tail. lam controls how strongly tail coverage is favored."""
    candidate_ids = np.asarray(candidate_ids)
    rel = _norm(rel_scores)
    remaining = list(range(len(candidate_ids)))
    selected = []
    n_tail = 0
    while remaining and len(selected) < N:
        frac_tail = (n_tail / len(selected)) if selected else 0.0
        need = max(0.0, p_tail - frac_tail)
        best, best_s = None, -1e9
        for pos in remaining:
            is_tail = not bool(is_head[candidate_ids[pos]])
            div = need if is_tail else 0.0
            s = (1 - lam) * rel[pos] + lam * div
            if s > best_s:
                best_s, best = s, pos
        selected.append(best)
        remaining.remove(best)
        if not bool(is_head[candidate_ids[best]]):
            n_tail += 1
    return candidate_ids[selected]
