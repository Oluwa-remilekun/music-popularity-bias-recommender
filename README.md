# Music Popularity-Bias Recommender

A comparative study of methods for reducing popularity bias in music recommender
systems. Collaborative-filtering recommenders tend to over-recommend already-popular
artists and under-serve niche ones; this project holds a base recommender fixed and
compares several mitigation methods to determine which best surfaces niche artists
without sacrificing recommendation quality, and how that trade-off varies across
listeners with different tastes.

## Approach

A fixed ALS (implicit-feedback matrix factorization) model generates candidate artists
per user. Four settings are compared, each re-ranking the same candidate lists:

1. **Baseline** — raw ALS with no correction (reference)
2. **Popularity penalty** — down-weights popular artists
3. **Binary xQuAD** — balances popular and niche groups
4. **Calibrated Popularity** — personalized re-ranking matched to each user's own taste

All settings are evaluated under one protocol: Recall@10 and NDCG@10 for accuracy, and
Average Recommendation Popularity, catalog coverage, and the Gini index for bias, with
significance testing and results broken down by listener group.

## Dataset

Last.fm-360K (~360,000 users, ~290,000 artists), with play counts treated as implicit
feedback. The dataset is downloaded at runtime and is not tracked in version control.

## Project structure

```
music-popularity-bias-recommender/
├── requirements.txt
├── data/                  # dataset (downloaded at runtime, untracked)
├── notebooks/
│   └── 01_explore_longtail.py   # data loading and long-tail analysis
├── src/
│   ├── data_loading.py    # loading, filtering, weighting, train/test split
│   ├── popularity.py      # artist popularity, head/tail split, user groups
│   ├── model.py           # ALS baseline and Most-Popular reference
│   ├── metrics.py         # accuracy and bias metrics
│   ├── rerank.py          # the three mitigation methods
│   └── evaluate.py        # experiment runner and comparison
├── app/
│   └── demo.py            # interactive comparison demo
└── paper/
    └── outline.md         # report structure
```


## Setup

pip install -r requirements.txt


Then open `notebooks/01_explore_longtail.py` and run the cells to load the dataset and
generate the long-tail popularity plot.

## Method notes

- All matrix operations use sparse representations.
- Evaluation uses a per-user train/test split; held-out interactions are excluded from
  training, and item popularity is computed on the training set only.
- Metrics are unit-tested against small hand-verified cases.
- The raw `plays` matrix is artists × users and is transposed for user-based recommendation.