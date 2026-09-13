"""Data loading, cleaning, and splitting for the Last.fm-360K dataset.

Roadmap: Weeks 1-2.  See Technical Spec, Section 2 (Data layer).

These functions are stubs. Fill in the TODOs yourself; the docstrings tell you
exactly what each one must do and flag the gotchas.
"""

from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
import scipy.sparse as sp
import numpy as np


def load_raw():
    """Download (first run only) and load the Last.fm-360K dataset.

    Returns
    -------
    artists : np.ndarray of str   -- artist name labels
    users   : np.ndarray of str   -- user id hashes
    plays   : scipy.sparse matrix -- shape (n_artists, n_users), play counts

    Orientation note
    ----------------
    `plays` is ARTISTS x USERS. For user-based recommendation you need
    USERS x ARTISTS, so you will transpose it (see `to_user_items`).
    Always sanity-check: plays.shape == (len(artists), len(users)).
    """
    artists, users, plays = get_lastfm()
    return artists, users, plays


def to_user_items(plays):
    """Return a USERS x ARTISTS CSR matrix from the raw plays matrix.

    TODO:
      - transpose `plays` and convert to CSR (.T.tocsr())
      - (recommended) assert the resulting shape is (n_users, n_artists)
    """
    raise NotImplementedError("Week 2: transpose plays to users x artists")


def filter_sparse(user_items, min_user_interactions=20, min_artist_listeners=20):
    """Drop very inactive users and very rarely-heard artists.

    Removing extreme-sparsity rows/cols cuts noise and shrinks the matrix.
    Keep everything SPARSE — do not densify.

    TODO:
      - count interactions per user and listeners per artist
      - keep only users/artists above the thresholds
      - return the filtered matrix (and, if useful, the kept index arrays)
    """
    raise NotImplementedError("Week 1-2: filter low-activity users/artists")


def weight_plays(user_items):
    """Down-weight huge play counts so a few superfans don't dominate.

    Hint: implicit provides `bm25_weight(matrix, K1=..., B=...)`.
    TODO: apply it and return the weighted CSR matrix (a near one-liner).
    """
    raise NotImplementedError("Week 2: apply bm25_weight")


def train_test_split(user_items, test_frac=0.2, seed=0):
    """Per-user hold-out split for evaluation.

    CRITICAL (leakage gotcha, Spec Section 11):
      - hold out `test_frac` of EACH user's interactions as the test set
      - REMOVE those held-out interactions from the training matrix
      - (later) popularity must be computed on the TRAINING matrix only

    TODO: implement and return
      train_matrix : users x artists CSR with test interactions removed
      test_dict    : {user_index: array/set of held-out artist indices}
    """
    raise NotImplementedError("Week 2: per-user train/test split, no leakage")
