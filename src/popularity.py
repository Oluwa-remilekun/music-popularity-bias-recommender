"""Artist popularity, head/tail split, user mainstream groups, and the long-tail figure."""

import numpy as np
import matplotlib.pyplot as plt


def artist_popularity(train_matrix):
    """Number of distinct listeners per artist, counted on the training matrix."""
    return np.asarray((train_matrix > 0).sum(axis=0)).ravel()


def head_tail_split(popularity, head_frac=0.2):
    """Label the top head_frac of artists by popularity as 'head' (popular); rest are tail."""
    popularity = np.asarray(popularity)
    n_head = max(1, int(len(popularity) * head_frac))
    top_idx = np.argsort(popularity)[::-1][:n_head]
    is_head = np.zeros(len(popularity), dtype=bool)
    is_head[top_idx] = True
    return is_head


def user_mainstream_groups(train_matrix, popularity, n_groups=3):
    """Group users by how mainstream their taste is (mean popularity of artists they play).
    Returns an array of group labels 0..n_groups-1 (0 = least mainstream)."""
    popularity = np.asarray(popularity)
    n_users = train_matrix.shape[0]
    mainstream = np.zeros(n_users)
    for u in range(n_users):
        arts = train_matrix[u].indices
        if len(arts) > 0:
            mainstream[u] = popularity[arts].mean()
    ranks = np.argsort(np.argsort(mainstream))
    labels = (ranks * n_groups // len(mainstream)).astype(int)
    labels[labels == n_groups] = n_groups - 1
    return labels


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
