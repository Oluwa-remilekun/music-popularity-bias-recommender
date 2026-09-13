"""Evaluation metrics: accuracy + popularity-bias.

Roadmap: Weeks 2-3.  See Technical Spec, Section 5.

>>> VALIDATE EVERY METRIC on a tiny hand-computed toy example (e.g. 5 users,
    6 artists) BEFORE running on the full data. A wrong metric does not crash;
    it silently gives plausible-but-wrong numbers for the whole project. <<<

Stubs only -- fill in the TODOs.
"""

import numpy as np


# ---------- accuracy (higher = better recommendations) ----------

def recall_at_k(recommended_ids, holdout_ids, k=10):
    """Of a user's held-out artists, how many appear in the top-k recommendations,
    as a fraction of the held-out set.
    TODO: implement for one user; average over users in the harness.
    """
    raise NotImplementedError("Week 2: recall@k")


def ndcg_at_k(recommended_ids, holdout_ids, k=10):
    """Like recall@k but rewards ranking correct items nearer the top.
    TODO: implement (use the standard DCG / ideal-DCG formulation).
    """
    raise NotImplementedError("Week 2: ndcg@k")


# ---------- popularity bias (this is where your contribution shows) ----------

def average_recommendation_popularity(recs, popularity):
    """Mean popularity of all recommended artists across users. Lower = less biased.
    `recs`: {user: [artist_ids]}.  TODO: implement.
    """
    raise NotImplementedError("Week 3: ARP")


def catalog_coverage(recs, n_artists):
    """Fraction of the whole catalog that appears in ANY user's recommendations.
    Higher = more of the long tail surfaces.  TODO: implement.
    """
    raise NotImplementedError("Week 3: catalog coverage")


def gini_index(recs, n_artists):
    """How unevenly recommendation exposure is spread across artists.
    0 = perfectly even, higher = concentrated on a few hits.  TODO: implement.
    """
    raise NotImplementedError("Week 3: Gini index")
