"""Experiment runner -- ties the whole pipeline together and produces the comparison.

Roadmap: Weeks 4-7.  See Technical Spec, Section 7.

This is the orchestration layer. It should REUSE the shared harness for every
method (build the harness once, add methods on top -- never edit the harness
per method, or comparisons stop being fair).

Stubs only -- fill in the TODOs.
"""

# from . import data_loading, popularity, model, metrics, rerank


def run_all_methods(train_matrix, test_dict, popularity, is_head,
                    user_groups, candidates, lam_by_method=None):
    """Score every setting on the SAME held-out data with the SAME metrics.

    Settings: raw ALS baseline, popularity penalty, Binary xQuAD,
    Calibrated Popularity (as each becomes available).

    TODO:
      - for each user, produce top-10 under each method (re-rank `candidates`)
      - compute accuracy (recall@10, ndcg@10) and bias (ARP, coverage, Gini)
      - return a results table: method -> {metric: value}
    """
    raise NotImplementedError("Weeks 4-6: run + score all methods")


def sweep_and_compare(train_matrix, test_dict, popularity, is_head,
                      user_groups, candidates, lam_values, seeds=(0, 1, 2)):
    """Vary each method's dial, repeat across seeds, test significance,
    and break results down by user mainstream group ('best for whom').

    TODO:
      - loop over lam_values and seeds, collect metrics
      - average across seeds (report mean +/- std)
      - run a paired significance test (e.g. Wilcoxon) before claiming a winner
      - produce the accuracy-vs-fairness trade-off figure and per-group tables
    """
    raise NotImplementedError("Weeks 5-7: sweep, seeds, significance, per-group")


if __name__ == "__main__":
    # Suggested end-to-end flow once the pieces are implemented:
    #   1. artists, users, plays = data_loading.load_raw()
    #   2. ui = data_loading.to_user_items(plays)
    #   3. ui = data_loading.filter_sparse(ui); ui = data_loading.weight_plays(ui)
    #   4. train, test = data_loading.train_test_split(ui)
    #   5. pop = popularity.artist_popularity(train)
    #   6. m = model.train_als(train); cand = model.recommend_all(m, train)
    #   7. run_all_methods(...) / sweep_and_compare(...)
    pass
