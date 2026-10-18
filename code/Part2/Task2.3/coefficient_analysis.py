"""Task 2.3: how much does each component contribute to the hybrid's final list?

Neither hybrid has one regression weight per model, but each has a parameter that
plays that role:

- FeatureCombination: the genre weight w scales content features against the
  collaborative item vectors (w = 0 is pure collaborative ItemKNN; tuned w = 0.1).
- Cascade: n_candidates sets the balance between the stages (n = 10 returns
  LightGCN's own top-10; large n lets SLIMElastic pick freely; tuned n = 100).

For every value we replay the saved Part 1.5 hybrid with only that parameter
changed, evaluate the list with the Task 2.1 evaluator, and measure where the
final items come from.

Run from the repository root (after export_data.py):
    python code/Part2/Task2.3/coefficient_analysis.py

Outputs in run_outputs/Part2/Task2.3/coefficient/:
    fc_genre_weight.csv, cascade_n_candidates.csv   one row per parameter value
    recommendations/, eval/                          lists and evaluator outputs
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_data import HYBRID_DIR, ROOT, LocalSavedHybrid, write_recommendations  # noqa: E402
from run_eval import K, evaluate, metric_columns, write_table  # noqa: E402
from run_hybrid import Experiment  # noqa: E402  (on sys.path via export_data)
from utils import top_items  # noqa: E402

OUT = ROOT / "run_outputs/Part2/Task2.3/coefficient"
GENRE_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]  # Part 1.4 grid plus the w = 0 ablation
N_CANDIDATES = [10, 20, 50, 100, 200, 500]       # every user has > 900 unseen items


def replay(experiment, name, params=None, state=None):
    """Recommendations of a saved hybrid with selected parameters overridden."""
    hybrid = LocalSavedHybrid(HYBRID_DIR / f"{name}.pth", experiment)
    checkpoint = dict(hybrid.checkpoint)
    checkpoint["params"] = {**checkpoint["params"], **(params or {})}
    checkpoint["fitted_state"] = {**checkpoint["fitted_state"], **(state or {})}
    hybrid.checkpoint = checkpoint
    return hybrid.recommend(split="test")


def overlap(lists, reference):
    """Mean fraction of each user's top-K that also appears in the reference top-K."""
    return sum(len(set(lists[u]) & set(reference[u][:K])) for u in lists) / (K * len(lists))


def full_ranking(experiment, model):
    """Each test user's ranking of all items, as {user: {item: rank}}.

    Seen items are masked to -inf and sort last; hybrids never recommend them.
    """
    data = experiment.data
    scores = experiment.scores[id(experiment.models[model])]
    rec = top_items(scores, data.test, data.n_items - 1)  # all items except padding
    users = data.dataset.id2token(data.dataset.uid_field, data.test.users.cpu().numpy())
    items = data.dataset.id2token(data.dataset.iid_field, rec.cpu().numpy())
    return {str(u): {item: rank for rank, item in enumerate(items[i], start=1)}
            for i, u in enumerate(users)}


def run(name, rec, label, config, extra):
    rec_dir, eval_dir = OUT / "recommendations", OUT / "eval"
    write_recommendations(label, rec, config, K, rec_dir)
    result = evaluate(rec_dir / f"{config['dataset']}_{label}_test_top{K}.json", eval_dir, label)
    return {**extra, **metric_columns(result)}


def feature_combination(experiment):
    config = experiment.data.config
    tuned = torch.load(HYBRID_DIR / "FeatureCombination.pth", map_location="cpu",
                       weights_only=True)["fitted_state"]["best_weight"]
    lists = {w: replay(experiment, "FeatureCombination", state={"best_weight": w})
             for w in GENRE_WEIGHTS}
    rows = []
    for w, rec in lists.items():
        extra = {
            "genre_weight": w,
            "tuned": w == tuned,
            # Share of the list that genre features changed relative to w = 0.
            "replaced_vs_no_genre": 1 - overlap(rec, lists[0.0]),
        }
        rows.append(run("FeatureCombination", rec, f"FC_w{w}", config, extra))
    write_table(rows, OUT / "fc_genre_weight.csv")


def cascade(experiment):
    config = experiment.data.config
    params = torch.load(HYBRID_DIR / "Cascade.pth", map_location="cpu",
                        weights_only=True)["params"]
    first, second = "LightGCN", "SLIMElastic"
    ranks = {model: full_ranking(experiment, model) for model in (first, second)}
    own_top = {model: {u: sorted(r, key=r.get)[:K] for u, r in ranks[model].items()}
               for model in ranks}
    rows = []
    for n in N_CANDIDATES:
        rec = replay(experiment, "Cascade", params={"n_candidates": n})
        items = [(u, item) for u, lst in rec.items() for item in lst]
        extra = {
            "n_candidates": n,
            "tuned": n == params["n_candidates"],
            # Where the final items come from.
            "overlap_LightGCN_top10": overlap(rec, own_top[first]),
            "overlap_SLIMElastic_top10": overlap(rec, own_top[second]),
            "mean_rank_in_LightGCN": sum(ranks[first][u][i] for u, i in items) / len(items),
            "mean_rank_in_SLIMElastic": sum(ranks[second][u][i] for u, i in items) / len(items),
        }
        rows.append(run("Cascade", rec, f"Cascade_n{n}", config, extra))
    write_table(rows, OUT / "cascade_n_candidates.csv")


def main():
    torch.set_num_threads(4)
    print("loading base models and scores ...")
    with redirect_stdout(io.StringIO()):
        experiment = Experiment()
    feature_combination(experiment)
    cascade(experiment)


if __name__ == "__main__":
    main()
