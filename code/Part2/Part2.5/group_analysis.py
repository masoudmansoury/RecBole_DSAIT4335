"""Part 2.5: accuracy and beyond-accuracy per user age group.

Run from the repository root, after code/Part2/Task2.3/export_data.py:
    python code/Part2/Part2.5/group_analysis.py

Outputs (run_outputs/Part2/Part2.5/):
    age_group_profile.tsv   per age group: number of users, mean train and test interactions
    per_user.tsv            every model's metrics for every test user, with age group
    age_group_means.tsv     model x age group: mean of each metric
    age_group_se.tsv        model x age group: standard error of each metric
    age_ndcg.png            nDCG@10 per model and age group
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/Part2/Part2.1"))

from evaluation.accuracy import per_user_accuracy  # noqa: E402
from evaluation.beyond_accuracy import aggregate_beyond_accuracy  # noqa: E402
from evaluation.data import (  # noqa: E402
    ground_truth_by_user,
    history_by_user,
    load_interactions,
    load_item_genres,
    load_recommendations,
    read_tsv,
)

REC_DIR = ROOT / "run_outputs/Part2/Task2.3/recommendations"
SPLIT_DIR = ROOT / "saved/splits"
DATA_DIR = ROOT / "dataset/ml-100k"
OUT = ROOT / "run_outputs/Part2/Part2.5"

K = 10
CALIBRATION_ALPHA = 0.01

MODELS = ["Random", "Pop", "UserKNN", "SLIMElastic", "EASE", "LightGCN",
          "Switching", "Cascade", "FeatureCombination"]
METRICS = ["ndcg", "recall", "precision", "hit", "mrr",
           "ild", "novelty", "miscalibration", "popularity_deviation"]

# Ages 1-24 -> "<25", 25-34 -> "25-34", 35-44 -> "35-44", 45+ -> "45+"
AGE_EDGES = [0, 24, 34, 44, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]


def load_data():
    """Test items and train history per user, the item catalog and item genres."""
    train = load_interactions(SPLIT_DIR / "ml-100k.train.tsv")
    valid = load_interactions(SPLIT_DIR / "ml-100k.valid.tsv")
    test = load_interactions(SPLIT_DIR / "ml-100k.test.tsv")

    ground_truth = ground_truth_by_user(test)
    histories = history_by_user(train)
    catalog = {row["item_id"] for row in train + valid + test}
    genres = load_item_genres(DATA_DIR / "ml-100k.item")
    return ground_truth, histories, catalog, genres


def load_users(ground_truth, histories):
    """One row per test user, with their age group and amount of data."""
    ages = {row["user_id"]: int(row["age"]) for row in read_tsv(DATA_DIR / "ml-100k.user")}

    users = pd.DataFrame({"user_id": list(ground_truth)})
    users["age_group"] = pd.cut(users["user_id"].map(ages), bins=AGE_EDGES, labels=AGE_LABELS)
    users["train_count"] = [len(histories.get(user, [])) for user in users["user_id"]]
    users["test_count"] = [len(ground_truth[user]) for user in users["user_id"]]
    return users


def evaluate_model(model, ground_truth, histories, catalog, genres):
    """One row per test user with all metrics of this model."""
    path = next(REC_DIR.glob(f"ml-100k_{model}_test_top*.json"))
    _, recommendations = load_recommendations(path, K)

    accuracy = per_user_accuracy(ground_truth, recommendations, K)
    _, beyond = aggregate_beyond_accuracy(
        recommendations, histories, genres, catalog,
        k=K, calibration_alpha=CALIBRATION_ALPHA,
    )
    return [{"model": model, "user_id": user, **accuracy[user], **beyond[user]}
            for user in accuracy]


def plot_ndcg(ndcg, ndcg_errors):
    ax = ndcg.plot.bar(yerr=ndcg_errors, capsize=2, figsize=(9, 4), rot=30)
    ax.set_xlabel("")
    ax.set_ylabel(f"nDCG@{K}")
    ax.legend(title="Age group")
    plt.tight_layout()
    plt.savefig(OUT / "age_ndcg.png", dpi=200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ground_truth, histories, catalog, genres = load_data()

    users = load_users(ground_truth, histories)
    profile = users.groupby("age_group", observed=True).agg(
        users=("user_id", "size"),
        train_count=("train_count", "mean"),
        test_count=("test_count", "mean"),
    )
    profile.to_csv(OUT / "age_group_profile.tsv", sep="\t")

    rows = []
    for model in MODELS:
        rows += evaluate_model(model, ground_truth, histories, catalog, genres)
    per_user = pd.DataFrame(rows).merge(users[["user_id", "age_group"]], on="user_id")
    per_user = per_user[["model", "user_id", "age_group", *METRICS]]
    per_user.to_csv(OUT / "per_user.tsv", sep="\t", index=False)

    groups = per_user.groupby(["model", "age_group"], observed=True)[METRICS]
    means = groups.mean().loc[MODELS]
    standard_errors = groups.sem().loc[MODELS]
    means.to_csv(OUT / "age_group_means.tsv", sep="\t")
    standard_errors.to_csv(OUT / "age_group_se.tsv", sep="\t")

    ndcg = means["ndcg"].unstack().loc[MODELS]
    print(ndcg.round(4))
    plot_ndcg(ndcg, standard_errors["ndcg"].unstack().loc[MODELS])


if __name__ == "__main__":
    main()