"""Evaluate recommendation accuracy for head, mid, and tail test items.

Run from the repository root:

    python code/Part2/Part2.5/item_group_evaluation.py

The script keeps the original recommendation ranking. For a group, it only
changes which held-out test items count as relevant.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PART21 = ROOT / "code" / "Part2" / "Part2.1"
sys.path.insert(0, str(PART21))

from evaluation.accuracy import aggregate_accuracy, user_accuracy  # noqa: E402
from evaluation.beyond_accuracy import item_popularity, mean_self_information  # noqa: E402
from evaluation.data import (  # noqa: E402
    history_by_user,
    load_interactions,
)


POPULARITY_FILE = ROOT / "run_outputs" / "Part2" / "Part2.5" / "item_popularity.tsv"
RECOMMENDATION_DIR = (
    ROOT / "run_outputs" / "Part2" / "Task2.3" / "recommendations"
)
TRAIN_FILE = ROOT / "saved" / "splits" / "ml-100k.train.tsv"
TEST_FILE = ROOT / "saved" / "splits" / "ml-100k.test.tsv"
OUTPUT_DIR = ROOT / "run_outputs" / "Part2" / "Part2.5"
K = 10
GROUPS = ("head", "mid", "tail")


def load_groups(path: str | Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required = {"item_id", "popularity_group"}
        if not required.issubset(rows.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        return {row["item_id"]: row["popularity_group"] for row in rows}


def load_recommendations(path: Path) -> tuple[str, dict[str, list[str]]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = payload.get("recommendations")
    if not isinstance(rows, dict):
        raise ValueError(f"{path} does not contain recommendations")
    recommendations = {}
    for user_id, row in rows.items():
        items = row.get("items") if isinstance(row, dict) else None
        if not isinstance(items, list) or len(items) < K:
            raise ValueError(f"{path} contains a user with fewer than {K} items")
        recommendations[str(user_id)] = [str(item) for item in items[:K]]
    return str(payload.get("model", path.stem)), recommendations


def grouped_accuracy(
    test_items: dict[str, list[str]],
    recommendations: dict[str, list[str]],
    item_groups: dict[str, str],
) -> list[dict[str, object]]:
    rows = []
    for group in GROUPS:
        group_test = {
            user_id: [item for item in items if item_groups.get(item) == group]
            for user_id, items in test_items.items()
        }
        evaluable = {
            user_id: items for user_id, items in group_test.items() if items
        }
        metrics = {
            user_id: user_accuracy(items, recommendations[user_id], K)
            for user_id, items in evaluable.items()
        }
        aggregate = aggregate_accuracy(metrics) if metrics else {
            name: 0.0
            for name in ("recall", "mrr", "ndcg")
        }
        rows.append(
            {
                "item_group": group,
                "users": len(evaluable),
                "test_interactions": sum(len(items) for items in evaluable.values()),
                "recall": round(aggregate["recall"], 6),
                "mrr": round(aggregate["mrr"], 6),
                "ndcg": round(aggregate["ndcg"], 6),
            }
        )
    return rows


def write_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "item_group",
        "users",
        "test_interactions",
        "recall",
        "mrr",
        "ndcg",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_group_beyond(path: Path, rows: list[dict[str, object]]) -> None:
    """Write beyond-accuracy values that have a meaningful item-group scope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "item_group",
        "recommendation_count",
        "recommendation_percent",
        "unique_items",
        "group_coverage",
        "novelty",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    item_groups = load_groups(POPULARITY_FILE)
    test_rows = load_interactions(TEST_FILE)
    train_rows = load_interactions(TRAIN_FILE)
    test_items = defaultdict(list)
    for row in test_rows:
        test_items[row["user_id"]].append(row["item_id"])
    histories = history_by_user(train_rows)

    accuracy_rows = []
    group_beyond_rows = []
    popularity = item_popularity(histories)
    catalog = set(item_groups)
    files = sorted(RECOMMENDATION_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No recommendation JSON files found in {RECOMMENDATION_DIR}")

    for path in files:
        model, recommendations = load_recommendations(path)
        accuracy_rows.extend(
            {"model": model, **row}
            for row in grouped_accuracy(test_items, recommendations, item_groups)
        )
        top_k_items = [
            item
            for items in recommendations.values()
            for item in items[:K]
        ]
        total_recommendations = len(top_k_items)
        for group in GROUPS:
            group_items = [
                item for item in top_k_items if item_groups.get(item) == group
            ]
            group_catalog = {
                item for item in catalog if item_groups.get(item) == group
            }
            group_beyond_rows.append(
                {
                    "model": model,
                    "item_group": group,
                    "recommendation_count": len(group_items),
                    "recommendation_percent": round(
                        100 * len(group_items) / total_recommendations, 6
                    ),
                    "unique_items": len(set(group_items)),
                    "group_coverage": round(
                        len(set(group_items)) / len(group_catalog), 6
                    ) if group_catalog else 0.0,
                    "novelty": round(
                        mean_self_information(group_items, popularity, len(catalog)),
                        6,
                    ),
                }
            )

    write_accuracy(OUTPUT_DIR / "item_group_accuracy.tsv", accuracy_rows)
    write_group_beyond(
        OUTPUT_DIR / "item_group_beyond_accuracy.tsv", group_beyond_rows
    )
    print(f"Saved grouped accuracy to {OUTPUT_DIR / 'item_group_accuracy.tsv'}")
    print(
        "Saved group-level beyond-accuracy results to "
        f"{OUTPUT_DIR / 'item_group_beyond_accuracy.tsv'}"
    )


if __name__ == "__main__":
    main()
