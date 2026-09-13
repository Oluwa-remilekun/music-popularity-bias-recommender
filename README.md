# Popularity-Bias Music Recommender

Senior seminar research project. **Goal:** hold a base recommender fixed and compare
several *popularity-bias mitigation* methods to find which best surfaces niche artists
without hurting recommendation quality — and for which listeners.

> **Note for anyone (or any agent) working in this repo:** the files below are
> **scaffolding** — real structure with function signatures, docstrings, and `TODO`
> markers, but the core logic is intentionally left unimplemented. This is a graded
> student project; the implementation and the learning belong to the student. Fill in
> the `TODO`s; don't replace them with a finished solution pulled from elsewhere.

## What the project compares
A fixed **ALS** recommender generates candidate artists per user. Four settings are
compared, all re-ranking the *same* candidates:
1. **Baseline** — raw ALS, no correction (reference)
2. **Popularity penalty** — down-weight popular artists (simplest)
3. **Binary xQuAD** — balance popular vs. niche groups (established method)
4. **Calibrated Popularity** — personalized, match each user's own taste mix (hardest)

Everything is judged by one shared harness: **Recall@10, NDCG@10** (accuracy) plus
**Average Recommendation Popularity, catalog coverage, Gini** (bias), with significance
testing and a breakdown by listener type.

## Folder layout
```
popularity-bias-recommender/
├── README.md              # you are here
├── requirements.txt       # dependencies (pip install -r requirements.txt)
├── .gitignore
├── data/                  # dataset downloads here (gitignored)
├── notebooks/
│   └── 01_explore_longtail.py   # Week 1: load data + draw the long-tail plot
├── src/
│   ├── data_loading.py    # load, filter, weight, train/test split
│   ├── popularity.py      # artist popularity, head/tail, user groups, Fig 1
│   ├── model.py           # ALS baseline + Most-Popular reference
│   ├── metrics.py         # accuracy + bias metrics
│   ├── rerank.py          # the 3 mitigation methods
│   └── evaluate.py        # experiment runner / comparison
├── app/
│   └── demo.py            # Streamlit comparison demo (Week 8, optional)
├── results/               # figures + tables land here
└── paper/
    └── outline.md         # report sections mapped to the plan
```

## Getting started (Week 1)
1. `pip install -r requirements.txt`
2. Open `notebooks/01_explore_longtail.py` (VS Code shows the `# %%` cells as a notebook).
3. Run the cells to load the data and confirm the shapes (~292k artists, ~359k users).
4. Complete the `TODO` to draw the long-tail plot — that's your first deliverable.

## Build order (matches the 8-week roadmap)
Weeks 1–3 build the foundation (data → ALS baseline → bias metrics). Weeks 4–6 add the
three mitigation methods **easy → hard** (penalty → xQuAD → Calibrated Popularity).
Week 7 is the paper; Week 8 is the demo + polish. The last two methods and the demo are
"reach" items — droppable without breaking the project.

## Key gotchas (read before coding)
- Keep matrices **sparse** — never convert the full user×artist matrix to dense.
- **No leakage:** held-out test interactions must be removed from training, and
  popularity must be computed on the **training set only**.
- **Validate each metric** on a tiny hand-computed toy example before trusting it at scale.
- Confirm matrix **orientation** (`plays` is artists×users; transpose for user-based recs).
