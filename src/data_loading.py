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
    """Return a users × artists matrix from the raw artists × users matrix."""
    user_items = plays.T.tocsr()
    return user_items


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
    """Split each user's interactions into train and test sets with no leakage."""
    rng = np.random.default_rng(seed)
    user_items = user_items.tocsr()
    train = user_items.copy().tolil()   # lil format is easy to edit
    test_dict = {}

    n_users = user_items.shape[0]
    for u in range(n_users):
        artist_ids = user_items[u].indices          # artists this user played
        if len(artist_ids) < 5:                      # skip very sparse users
            continue
        n_test = max(1, int(len(artist_ids) * test_frac))
        test_ids = rng.choice(artist_ids, size=n_test, replace=False)
        test_dict[u] = test_ids
        train[u, test_ids] = 0                       # remove test artists from train

    train = train.tocsr()
    train.eliminate_zeros()
    return train, test_dict
