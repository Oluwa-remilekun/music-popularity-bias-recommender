"""Artist popularity, head/tail split, user mainstream groups, and Figure 1.

Roadmap: Weeks 1 and 3.  See Technical Spec, Sections 3 & 5.
Stubs only -- fill in the TODOs.
"""

import numpy as np
import matplotlib.pyplot as plt


def artist_popularity(train_matrix):
    """Number of distinct listeners per artist, counted on the training matrix."""
    listeners = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
    return listeners


def head_tail_split(popularity, head_frac=0.2):
    """Label artists as 'head' (popular) vs 'tail' (niche).

    TODO: mark the top `head_frac` of artists by popularity as head.
    Return a boolean mask `is_head` (True = popular head).
    """
    raise NotImplementedError("Week 3: split catalog into head vs tail")


def long_tail_plot(popularity, save_path="results/fig1_longtail.png"):
    """Plot artists ranked by popularity to show the long-tail distribution."""
    sorted_counts = np.sort(popularity)[::-1]
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_counts)
    plt.yscale("log")
    plt.xlabel("Artist rank (most to least popular)")
    plt.ylabel("Number of listeners (log scale)")
    plt.title("Long-tail distribution of artist popularity")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


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
