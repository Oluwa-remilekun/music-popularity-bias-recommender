"""Baseline recommenders: ALS (main) and Most-Popular (reference).

Roadmap: Week 2.  See Technical Spec, Section 4.
Stubs only -- fill in the TODOs.
"""

from implicit.als import AlternatingLeastSquares
import numpy as np


def train_als(train_matrix, factors=64, regularization=0.05, iterations=15,
              random_state=0):
    """Train the ALS implicit-feedback matrix-factorization baseline.

    This is the ONE machine-learning model you need to understand conceptually:
    it learns a short 'taste vector' for each user and artist so their dot
    product predicts preference. The library does the math.

    Hint:
        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        random_state=random_state)
        model.fit(train_matrix)   # train_matrix: users x artists (weighted)

    TODO: create, fit, and return the model.
    """
    raise NotImplementedError("Week 2: train the ALS baseline")


def recommend_all(model, train_matrix, N=100):
    """Generate a top-N CANDIDATE list per user (already-played excluded).

    Use a LARGE N here (e.g. 100). Your re-rankers reorder these candidates;
    the final recommendations you evaluate are the top 10 after re-ranking.

    Hint (per user):
        ids, scores = model.recommend(userid, train_matrix[userid],
                                      N=N, filter_already_liked_items=True)

    TODO: loop or batch over users and return
        {user_index: (candidate_artist_ids, candidate_scores)}
    """
    raise NotImplementedError("Week 2: generate candidate lists for all users")


def most_popular_baseline(popularity, N=10):
    """Trivial reference recommender: the globally most-popular artists for everyone.

    It ignores personal taste and is your 'worst-case popularity bias' floor.
    TODO: return the top-N artist ids by popularity.
    """
    raise NotImplementedError("Week 3: most-popular reference recommender")
