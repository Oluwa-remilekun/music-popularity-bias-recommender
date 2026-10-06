"""Three-way comparison: baseline vs popularity-penalty vs Binary xQuAD."""
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
is_head = popularity.head_tail_split(pop, head_frac=0.2)
n_artists = train.shape[1]
print(f"data+split ready ({time.time()-t0:.0f}s)")

tw = bm25_weight(train, K1=100, B=0.8).tocsr()
als = model.train_als(tw, factors=64, iterations=12, random_state=0)
print(f"ALS trained ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(1)
sample = rng.choice(np.array(sorted(test_dict.keys())), size=3000, replace=False)
cand = model.recommend_all(als, train, sample, N=80)
# each user's fraction of tail (niche) artists in their training profile, for xQuAD
p_tail = {}
for u in sample:
    arts = train[int(u)].indices
    p_tail[int(u)] = float(np.mean(~is_head[arts])) if len(arts) else 0.5
print(f"candidates ready ({time.time()-t0:.0f}s)")

def score(recs):
    R = np.mean([metrics.recall_at_k(recs[int(u)], test_dict[int(u)]) for u in sample])
    N_ = np.mean([metrics.ndcg_at_k(recs[int(u)], test_dict[int(u)]) for u in sample])
    return dict(recall=round(float(R),4), ndcg=round(float(N_),4),
                arp=round(metrics.average_recommendation_popularity(recs,pop),1),
                coverage=round(metrics.catalog_coverage(recs,n_artists),5),
                gini=round(metrics.gini_index(recs,n_artists),4))

lams = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
out = {"penalty": {}, "xquad": {}}
for lam in lams:
    recs_p, recs_x = {}, {}
    for u in sample:
        ids, sc = cand[int(u)]
        recs_p[int(u)] = list(rerank.rerank_popularity_penalty(ids, sc, pop, lam=lam, N=10))
        recs_x[int(u)] = list(rerank.rerank_binary_xquad(ids, sc, is_head, p_tail[int(u)], lam=lam, N=10))
    out["penalty"][lam] = score(recs_p)
    out["xquad"][lam]   = score(recs_x)
    print(f"lam={lam}  penalty R={out['penalty'][lam]['recall']} ARP={out['penalty'][lam]['arp']}  "
          f"|  xquad R={out['xquad'][lam]['recall']} ARP={out['xquad'][lam]['arp']}")

json.dump(out, open("results/comparison.json","w"), indent=2)

plt.figure(figsize=(7.5,4.6))
for name, color, lab in [("penalty","#1C7293","Popularity penalty"), ("xquad","#E8A33D","Binary xQuAD")]:
    rc = [out[name][l]["recall"] for l in lams]
    ap = [out[name][l]["arp"] for l in lams]
    plt.plot(ap, rc, "-o", color=color, label=lab)
    for l,x,y in zip(lams,ap,rc): plt.annotate(f"{l}", (x,y), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.gca().invert_xaxis()
plt.xlabel("Average Recommendation Popularity (lower = less biased →)")
plt.ylabel("Recall@10 (higher = more accurate)")
plt.title("Popularity penalty vs. Binary xQuAD: accuracy–fairness trade-off")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig3_comparison.png", dpi=150, bbox_inches="tight")
print(f"DONE ({time.time()-t0:.0f}s) -> results/comparison.json + results/fig3_comparison.png")
