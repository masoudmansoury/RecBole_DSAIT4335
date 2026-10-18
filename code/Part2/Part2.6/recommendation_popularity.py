"""Compare the popularity of recommendations made by each model.

The script uses the existing Task 2.3 recommendation files. Every model is
evaluated with its first 10 recommendations per user, so top-100 and top-10
files are compared fairly.

Run from the repository root:

    python code/Part2/Part2.6/recommendation_popularity.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POPULARITY_FILE = ROOT / "run_outputs" / "Part2" / "Part2.5" / "item_popularity.tsv"
RECOMMENDATION_DIR = (
    ROOT / "run_outputs" / "Part2" / "Task2.3" / "recommendations"
)
OUTPUT_DIR = ROOT / "run_outputs" / "Part2" / "Part2.6"
K = 10
GROUPS = ("head", "mid", "tail")


def load_popularity(path: str | Path) -> dict[str, dict[str, object]]:
    """Load item popularity and group information from the Part 2.5 table."""
    with open(path, encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required = {
            "item_id",
            "interaction_count",
            "popularity_rank",
            "popularity_group",
        }
        if not required.issubset(rows.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        return {
            row["item_id"]: {
                "interaction_count": int(row["interaction_count"]),
                "popularity_rank": int(row["popularity_rank"]),
                "popularity_group": row["popularity_group"],
            }
            for row in rows
        }


def load_recommendations(
    path: str | Path, popularity: dict[str, dict[str, object]]
) -> tuple[str, Counter[str], list[str]]:
    """Load one model and count its head/mid/tail recommendations."""
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)

    model = str(payload.get("model", Path(path).stem))
    rows = payload.get("recommendations")
    if not isinstance(rows, dict):
        raise ValueError(f"{path} does not contain recommendations")

    groups: Counter[str] = Counter()
    recommended_items: list[str] = []
    for row in rows.values():
        items = row.get("items") if isinstance(row, dict) else None
        if not isinstance(items, list) or len(items) < K:
            raise ValueError(f"{path} contains a user with fewer than {K} items")
        for item in items[:K]:
            item_id = str(item)
            if item_id not in popularity:
                raise ValueError(f"{path} contains unknown item {item_id}")
            group = str(popularity[item_id]["popularity_group"])
            if group not in GROUPS:
                raise ValueError(f"unknown popularity group {group} for {item_id}")
            groups[group] += 1
            recommended_items.append(item_id)
    return model, groups, recommended_items


def write_results(
    path: str | Path, rows: list[dict[str, object]]
) -> None:
    """Write model-level popularity exposure as a TSV file."""
    fieldnames = [
        "model",
        "users",
        "recommendations",
        "head_count",
        "mid_count",
        "tail_count",
        "head_percent",
        "mid_percent",
        "tail_percent",
        "unique_items",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write a stacked bar chart showing head/mid/tail recommendation exposure."""
    import matplotlib.pyplot as plt

    models = [str(row["model"]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 6))
    bottom = [0.0] * len(rows)
    colors = {"head": "steelblue", "mid": "darkorange", "tail": "seagreen"}

    for group in GROUPS:
        values = [float(row[f"{group}_percent"]) for row in rows]
        axis.bar(models, values, bottom=bottom, label=group, color=colors[group])
        bottom = [current + value for current, value in zip(bottom, values)]

    axis.set_title("Popularity of recommended movies")
    axis.set_xlabel("Recommendation model")
    axis.set_ylabel("Percentage of top-10 recommendations")
    axis.set_ylim(0, 100)
    axis.legend(title="Movie popularity group")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    popularity = load_popularity(POPULARITY_FILE)
    results: list[dict[str, object]] = []

    recommendation_files = sorted(RECOMMENDATION_DIR.glob("*.json"))
    if not recommendation_files:
        raise FileNotFoundError(f"No recommendation JSON files found in {RECOMMENDATION_DIR}")

    for path in recommendation_files:
        model, groups, recommended_items = load_recommendations(path, popularity)
        total = sum(groups.values())
        user_count = len(json.loads(path.read_text(encoding="utf-8"))["recommendations"])
        results.append(
            {
                "model": model,
                "users": user_count,
                "recommendations": total,
                **{f"{group}_count": groups[group] for group in GROUPS},
                **{
                    f"{group}_percent": round(100 * groups[group] / total, 2)
                    for group in GROUPS
                },
                "unique_items": len(set(recommended_items)),
            }
        )

    results.sort(key=lambda row: str(row["model"]).lower())
    write_results(OUTPUT_DIR / "recommendation_popularity.tsv", results)
    write_plot(OUTPUT_DIR / "recommendation_popularity.png", results)
    print(f"Analysed {len(results)} models using the first {K} items per user.")
    print(f"Saved results to {OUTPUT_DIR / 'recommendation_popularity.tsv'}")
    print(f"Saved plot to {OUTPUT_DIR / 'recommendation_popularity.png'}")


if __name__ == "__main__":
    main()
