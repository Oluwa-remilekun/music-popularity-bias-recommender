# Report Outline

Draft sections incrementally as you build (Weeks 1, 3, then fully in Week 7),
so the writing rides on finished results instead of a last-minute scramble.

## 1. Introduction
- The problem: popularity bias in music recommendation.
- Why it matters: discovery, long-tail artists, unfair to non-mainstream listeners.
- Research question: which mitigation method best surfaces niche artists without
  hurting quality, and does the best method depend on the listener?
- (Reuse / adapt your approved proposal here.)

## 2. Related Work
- The anchor paper (Kowald, Schedl & Lex) and its popularity-bias findings.
- The methods you compare: xQuAD; Calibrated Popularity.
- Draft a stub in Week 1 while the reading is fresh.

## 3. Data
- Last.fm-360K: size, play counts as implicit feedback, how you filter/weight it.
- State your definitions once (how "popularity" and head/tail are measured).

## 4. Method
- Fixed ALS base model + candidate generation.
- The three re-rankers: popularity penalty, Binary xQuAD, Calibrated Popularity.
- The shared evaluation harness (split, metrics, seeds, significance).

## 5. Experiments & Results
- Baseline characterization (bias exists; worse for low-mainstream users).
- Method comparison: accuracy-vs-fairness trade-off figure.
- Per-user-group breakdown ("best for whom").

## 6. Discussion
- What the comparison shows (likely a *conditional* result -- which method wins where).
- Limitations (single dataset, artist-level, no timestamps, evaluator caveats).

## 7. Conclusion & Future Work
- The answer to the research question.
- Future work: Calibrated Popularity if dropped, k-fold, more/newer datasets.

## Deliverables checklist
- [ ] Written report (this document, filled in)
- [ ] Code + result figures (this repo)
- [ ] Interactive comparison demo (app/demo.py)
