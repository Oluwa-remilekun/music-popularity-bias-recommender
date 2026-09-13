"""Artist popularity, head/tail split, user mainstream groups, and Figure 1.

Roadmap: Weeks 1 and 3.  See Technical Spec, Sections 3 & 5.
Stubs only -- fill in the TODOs.
"""

import numpy as np
import matplotlib.pyplot as plt


def artist_popularity(train_matrix):
    """Popularity of each artist = number of distinct users who played it.

    IMPORTANT: compute this on the TRAINING matrix only (no test leakage).

    TODO: from the users x artists training matrix, count nonzero entries
    per artist (column). Return an np.ndarray of length n_artists.
    """
    raise NotImplementedError("Week 3: count listeners per artist on TRAIN")


def head_tail_split(popularity, head_frac=0.2):
    """Label artists as 'head' (popular) vs 'tail' (niche).

    TODO: mark the top `head_frac` of artists by popularity as head.
    Return a boolean mask `is_head` (True = popular head).
    """
    raise NotImplementedError("Week 3: split catalog into head vs tail")


def long_tail_plot(popularity, save_path="results/fig1_longtail.png"):
    """Draw the rank-vs-popularity long-tail curve (your Week 1 Figure 1).

    TODO:
      - sort popularity in descending order
      - plot rank (x) vs popularity (y); a log scale on y (or both) shows the
        classic 'playground slide' shape
      - label axes, title it, and save to `save_path`
    This figure is the visual proof that popularity bias exists in your data.
    """
    raise NotImplementedError("Week 1: draw + save the long-tail plot")


def user_mainstream_groups(train_matrix, popularity, n_groups=3):
    """Segment users into low / medium / high 'mainstream-ness'.

    A user's mainstream-ness = the average popularity of the artists they
    listen to. Low-mainstream users are the ones popularity bias hurts most,
    and the ones your 'best for whom' analysis focuses on.

    TODO:
      - for each user, compute the mean popularity of the artists in their row
      - bucket users into `n_groups` (e.g. by tertiles)
      - return an np.ndarray of group labels per user
    """
    raise NotImplementedError("Week 3: bucket users by taste mainstream-ness")
