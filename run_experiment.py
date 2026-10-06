"""Full baseline + popularity-penalty experiment. Produces results/ numbers and the trade-off curve."""
import time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from implicit.nearest_neighbours import bm25_weight
from src import data_loading, popularity, model, metrics, rerank

t0 = time.time()
artists, users, plays = data_loading.load_raw()
ui = data_loading.to_user_items(plays)
train, test_dict = data_loading.train_test_split(ui, test_frac=0.2, seed=0)
pop = popularity.artist_popularity(train).astype(float)
n_artists = train.shape[1]
print(f"data+split ready ({time.time()-t0:.0f}s)")

tw = bm25_weight(train, K1=100, B=0.8).tocsr()
als = model.train_als(tw, factors=64, iterations=12, random_state=0)
print(f"ALS trained ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(1)
sample = rng.choice(np.array(sorted(test_dict.keys())), size=3000, replace=False)
cand = model.recommend_all(als, train, sample, N=80)
print(f"candidates ready ({time.time()-t0:.0f}s)")

lams = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
results = {}
for lam in lams:
    recs = {}
    for u in sample:
        ids, sc = cand[int(u)]
        recs[int(u)] = list(rerank.rerank_popularity_penalty(ids, sc, pop, lam=lam, N=10))
    R = np.mean([metrics.recall_at_k(recs[int(u)], test_dict[int(u)]) for u in sample])
    N_ = np.mean([metrics.ndcg_at_k(recs[int(u)], test_dict[int(u)]) for u in sample])
    arp = metrics.average_recommendation_popularity(recs, pop)
    cov = metrics.catalog_coverage(recs, n_artists)
    gini = metrics.gini_index(recs, n_artists)
    results[lam] = dict(recall=round(float(R),4), ndcg=round(float(N_),4),
                        arp=round(arp,1), coverage=round(cov,5), gini=round(gini,4))
    print(f"lam={lam}: {results[lam]}")

json.dump(results, open("results/penalty_sweep.json","w"), indent=2)

# trade-off curve
rc = [results[l]["recall"] for l in lams]
ap = [results[l]["arp"] for l in lams]
plt.figure(figsize=(7,4.4))
plt.plot(ap, rc, "-o")
for l,x,y in zip(lams,ap,rc): plt.annotate(f"λ={l}", (x,y), textcoords="offset points", xytext=(6,6))
plt.gca().invert_xaxis()
plt.xlabel("Average Recommendation Popularity (lower = less biased →)")
plt.ylabel("Recall@10 (higher = more accurate)")
plt.title("Accuracy vs. fairness trade-off (popularity penalty)")
plt.tight_layout()
plt.savefig("results/fig2_tradeoff.png", dpi=150, bbox_inches="tight")
print(f"DONE ({time.time()-t0:.0f}s) -> results/penalty_sweep.json + results/fig2_tradeoff.png")
