"""
score_from_saved.py

Recomputes Recall@K, Precision@K and F1@K directly from the JSON files
produced by save_recommendations.py and the ground-truth test split
produced by save_split.py -- without re-loading any RecBole model.

This lets you compare several models' recommendation lists on the exact
same held-out test interactions, as long as every recommendation file
was produced from configs that share the same dataset, seed, and
eval_args (see the README for how to guarantee that).

Usage:
    python score_from_saved.py \
        --test_tsv saved/splits/ml-100k.test.tsv \
        --rec_json saved/recommendations/ml-100k_BPR_top10.json \
                    saved/recommendations/ml-100k_LightGCN_top10.json \
        --k 10
"""
import argparse
import json
from collections import defaultdict

import pandas as pd


def load_ground_truth(test_tsv):
    df = pd.read_csv(test_tsv, sep="\t", dtype=str)
    uid_col, iid_col = df.columns[0], df.columns[1]
    gt = defaultdict(set)
    for u, i in zip(df[uid_col], df[iid_col]):
        gt[u].add(i)
    return gt


def score_one_model(rec_json_path, ground_truth, k):
    with open(rec_json_path) as f:
        payload = json.load(f)

    recall_sum, precision_sum, f1_sum, n_users = 0.0, 0.0, 0.0, 0

    for user_token, rec in payload["recommendations"].items():
        pos_items = ground_truth.get(user_token)
        if not pos_items:
            continue  # user has no held-out interactions to score against
        topk_items = rec["items"][:k]
        n_hit = len(set(topk_items) & pos_items)

        precision = n_hit / k
        recall = n_hit / len(pos_items)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        recall_sum += recall
        precision_sum += precision
        f1_sum += f1
        n_users += 1

    return {
        "model": payload["model"],
        "n_users_scored": n_users,
        f"recall@{k}": round(recall_sum / n_users, 4),
        f"precision@{k}": round(precision_sum / n_users, 4),
        f"f1@{k}": round(f1_sum / n_users, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tsv", type=str, required=True)
    parser.add_argument("--rec_json", type=str, nargs="+", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.test_tsv)

    rows = []
    for path in args.rec_json:
        rows.append(score_one_model(path, ground_truth, args.k))

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
