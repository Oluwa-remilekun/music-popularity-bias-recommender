"""Baseline recommenders: ALS (main) and Most-Popular (reference)."""

from implicit.als import AlternatingLeastSquares
import numpy as np


def train_als(train_matrix, factors=64, regularization=0.05, iterations=15, random_state=0):
    """Train the ALS implicit-feedback matrix-factorization baseline."""
    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=random_state)
    model.fit(train_matrix)
    return model


def recommend_all(model, train_matrix, userids, N=100):
    """Generate a top-N candidate list per user. Returns {user: (artist_ids, scores)}."""
    cand = {}
    for u in userids:
        ids, scores = model.recommend(int(u), train_matrix[int(u)], N=N,
                                      filter_already_liked_items=True)
        cand[int(u)] = (np.asarray(ids), np.asarray(scores, dtype=float))
    return cand


def most_popular_baseline(popularity, N=10):
    """Trivial reference: the globally most-popular artists, for everyone."""
    return np.argsort(np.asarray(popularity))[::-1][:N]
