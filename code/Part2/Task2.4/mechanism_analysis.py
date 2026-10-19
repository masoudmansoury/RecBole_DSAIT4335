"""Task 2.4: explain the behaviour found in the Task 2.3 coefficient analysis.

Cascade: LightGCN only replaces ~2% of SLIMElastic's top-10 at the tuned
n_candidates, yet catalog coverage drops ~11%. We characterise the items
LightGCN vetoes (and the replacements it lets in): how popular they are,
how many users SLIMElastic showed them to, and whether they were relevant.

FeatureCombination: as the genre weight grows, miscalibration first falls and
then rises sharply. We measure how strongly each list concentrates on the
user's dominant history genre, and how much of the user's genre profile
disappears from the list.

Run from the repository root (after the Task 2.3 scripts; standard library only):
    python code/Part2/Task2.4/mechanism_analysis.py

Outputs in run_outputs/Part2/Task2.4/:
    cascade_component_popularity.csv   popularity profile of each stage's own top-10
    cascade_veto_by_n.csv              vetoed vs. introduced items per n_candidates
    fc_genre_concentration.csv         genre concentration per genre weight
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/Part2/Part2.1"))

# Reuse the Task 2.1 definitions so popularity and genre shares match the metrics.
from evaluation.beyond_accuracy import genre_distribution, item_popularity, popularity_groups  # noqa: E402
from evaluation.data import (  # noqa: E402
    ground_truth_by_user,
    history_by_user,
    load_interactions,
    load_item_genres,
    load_recommendations,
)

SPLITS = ROOT / "saved/splits"
T23 = ROOT / "run_outputs/Part2/Task2.3"
REC = T23 / "recommendations"
SWEEP = T23 / "coefficient"
OUT = ROOT / "run_outputs/Part2/Task2.4"
K = 10
N_CANDIDATES = [10, 20, 50, 100, 200, 500]
GENRE_WEIGHTS = ["0.0", "0.1", "0.25", "0.5", "1.0", "2.0"]


def recs(path):
    return load_recommendations(path, K)[1]


def mean_or_none(values):
    values = list(values)
    return fmean(values) if values else None


def write_table(rows, name):
    path = OUT / name
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 4) if isinstance(value, float) else value
                             for key, value in row.items()})
    print(f"saved {path.relative_to(ROOT)}")


def sweep_metrics(table, key):
    """Evaluator metrics from the Task 2.3 sweep table, indexed by parameter value."""
    with open(SWEEP / table, encoding="utf-8") as f:
        return {row[key]: row for row in csv.DictReader(f)}


class Context:
    def __init__(self):
        train = load_interactions(SPLITS / "ml-100k.train.tsv")
        valid = load_interactions(SPLITS / "ml-100k.valid.tsv")
        test = load_interactions(SPLITS / "ml-100k.test.tsv")
        self.history = history_by_user(train)
        self.truth = ground_truth_by_user(test)
        self.genres = load_item_genres(ROOT / "dataset/ml-100k/ml-100k.item")
        # Same popularity and head/mid/tail tertiles as the Task 2.1 evaluator.
        self.popularity = item_popularity(self.history)
        catalog = {row["item_id"] for row in [*train, *valid, *test]}
        self.tier = popularity_groups(self.popularity, catalog)

    def item_profile(self, pairs, prefix):
        """Popularity and relevance of (user, item) recommendation slots."""
        pairs = list(pairs)
        return {
            f"{prefix}_slots": len(pairs),
            f"{prefix}_mean_train_popularity": mean_or_none(self.popularity[i] for _, i in pairs),
            f"{prefix}_tail_share": mean_or_none(self.tier[i] == "tail" for _, i in pairs),
            f"{prefix}_head_share": mean_or_none(self.tier[i] == "head" for _, i in pairs),
            f"{prefix}_hit_rate": mean_or_none(i in self.truth.get(u, ()) for u, i in pairs),
        }


def cascade(ctx):
    slim = {u: items[:K] for u, items in recs(REC / "ml-100k_SLIMElastic_test_top100.json").items()}
    lightgcn = {u: items[:K] for u, items in recs(REC / "ml-100k_LightGCN_test_top100.json").items()}
    metrics = sweep_metrics("cascade_n_candidates.csv", "n_candidates")
    slim_exposure = Counter(i for items in slim.values() for i in items)

    def slots(lists):
        return [(u, i) for u, items in lists.items() for i in items]

    tuned = recs(SWEEP / "recommendations/ml-100k_Cascade_n100_test_top10.json")
    rows = []
    for name, lists in [("LightGCN", lightgcn), ("SLIMElastic", slim), ("Cascade (n=100)", tuned)]:
        rows.append({"model": name,
                     "distinct_items": len({i for _, i in slots(lists)}),
                     **ctx.item_profile(slots(lists), "top10")})
    write_table(rows, "cascade_component_popularity.csv")

    rows = []
    for n in N_CANDIDATES:
        final = recs(SWEEP / f"recommendations/ml-100k_Cascade_n{n}_test_top10.json")
        vetoed = [(u, i) for u in slim for i in slim[u] if i not in final[u]]
        introduced = [(u, i) for u in final for i in final[u] if i not in slim[u]]
        kept = [(u, i) for u in final for i in final[u] if i in slim[u]]
        # Items that vanish from (or newly enter) every user's list.
        slim_items = set(slim_exposure)
        final_items = {i for items in final.values() for i in items}
        lost, gained = slim_items - final_items, final_items - slim_items
        rows.append({
            "n_candidates": n,
            "vetoed_share": len(vetoed) / (K * len(slim)),
            **ctx.item_profile(vetoed, "vetoed"),
            "vetoed_median_slim_exposure": median(slim_exposure[i] for _, i in vetoed) if vetoed else None,
            **ctx.item_profile(introduced, "introduced"),
            **ctx.item_profile(kept, "kept"),
            "catalog_items_lost": len(lost),
            "catalog_items_gained": len(gained),
            "lost_items_shown_to_one_user_share": mean_or_none(slim_exposure[i] == 1 for i in lost),
            "catalog_coverage": float(metrics[str(n)]["catalog_coverage"]),
            "ndcg@10": float(metrics[str(n)]["ndcg@10"]),
        })
    write_table(rows, "cascade_veto_by_n.csv")


def fc(ctx):
    metrics = sweep_metrics("fc_genre_weight.csv", "genre_weight")
    lists = {w: recs(SWEEP / f"recommendations/ml-100k_FC_w{w}_test_top10.json") for w in GENRE_WEIGHTS}
    base = lists["0.0"]

    # Each user's genre profile, computed exactly as the miscalibration metric does.
    profile = {u: genre_distribution(ctx.history.get(u, ()), ctx.genres) for u in base}
    top_genre = {u: max(p, key=p.get) for u, p in profile.items() if p}

    rows = []
    for w in GENRE_WEIGHTS:
        rec = lists[w]
        top_share_hist, top_share_rec, n_genres, missing = [], [], [], []
        for u, top in top_genre.items():
            dist = genre_distribution(rec[u], ctx.genres)
            top_share_hist.append(profile[u][top])
            top_share_rec.append(dist.get(top, 0.0))
            n_genres.append(len(dist))
            # Probability mass of the user's profile that the list does not cover at all.
            missing.append(sum(p for g, p in profile[u].items() if g not in dist))
        introduced = [(u, i) for u in rec for i in rec[u] if i not in base[u]]
        removed = [(u, i) for u in base for i in base[u] if i not in rec[u]]
        rows.append({
            "genre_weight": float(w),
            "top_genre_share_history": fmean(top_share_hist),
            "top_genre_share_recs": fmean(top_share_rec),
            "distinct_genres_in_list": fmean(n_genres),
            "profile_mass_missing_from_list": fmean(missing),
            **ctx.item_profile(introduced, "introduced"),
            **ctx.item_profile(removed, "removed"),
            "miscalibration": float(metrics[w]["miscalibration"]),
            "ild": float(metrics[w]["ild"]),
            "novelty": float(metrics[w]["novelty"]),
            "catalog_coverage": float(metrics[w]["catalog_coverage"]),
            "ndcg@10": float(metrics[w]["ndcg@10"]),
        })
    write_table(rows, "fc_genre_concentration.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    cascade(ctx)
    fc(ctx)


if __name__ == "__main__":
    main()
