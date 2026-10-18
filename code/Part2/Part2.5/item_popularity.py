"""Calculate item popularity and assign simple popularity groups.

Popularity is calculated from interactions that are available before testing.

"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "saved" / "splits" / "ml-100k.train.tsv"
DEFAULT_OUTPUT = ROOT / "run_outputs" / "Part2" / "Part2.5" / "item_popularity.tsv"
DEFAULT_PLOT = ROOT / "run_outputs" / "Part2" / "Part2.5" / "item_popularity.png"


def _field_name(field: str) -> str:
    """Remove RecBole's type suffix, for example ``item_id:token``."""
    return field.split(":", 1)[0]


def count_item_interactions(path: str | Path) -> Counter[str]:
    """Count training interactions for every item in a TSV interaction file."""
    counts: Counter[str] = Counter()
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        fields = {_field_name(field): field for field in reader.fieldnames}
        if "item_id" not in fields:
            raise ValueError(f"{path} must contain an item_id column")

        for row in reader:
            item_id = row[fields["item_id"]]
            if item_id:
                counts[item_id] += 1
    return counts


def popularity_table(counts: Counter[str]) -> list[dict[str, object]]:
    """Return items ranked by count, with deterministic head/mid/tail groups.

    The groups contain approximately one third of the items each. Ties are
    ordered by item ID so repeated runs produce the same result.
    """
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    item_count = len(ordered)
    table: list[dict[str, object]] = []

    for index, (item_id, interaction_count) in enumerate(ordered, start=1):
        group_index = min((3 * (index - 1)) // max(item_count, 1), 2)
        group = ("head", "mid", "tail")[group_index]
        table.append(
            {
                "item_id": item_id,
                "interaction_count": interaction_count,
                "popularity_rank": index,
                "popularity_group": group,
            }
        )
    return table


def write_table(path: str | Path, table: list[dict[str, object]]) -> None:
    """Write the popularity table as a TSV file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "interaction_count",
        "popularity_rank",
        "popularity_group",
    ]
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(table)


def write_plot(path: str | Path, table: list[dict[str, object]]) -> None:
    """Write a descending popularity curve, with rank on the x-axis."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranks = [int(row["popularity_rank"]) for row in table]
    counts = [int(row["interaction_count"]) for row in table]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(ranks, counts, color="steelblue", linewidth=1.5)
    axis.set_title("Movie popularity distribution")
    axis.set_xlabel("Movie popularity rank (1 = most popular)")
    axis.set_ylabel("Number of interactions")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    counts = count_item_interactions(DEFAULT_INPUT)
    table = popularity_table(counts)
    write_table(DEFAULT_OUTPUT, table)
    write_plot(DEFAULT_PLOT, table)

    group_counts = Counter(row["popularity_group"] for row in table)
    print(f"Read {sum(counts.values())} interactions for {len(table)} items.")
    print(
        "Groups: "
        + ", ".join(
            f"{group}={group_counts[group]}"
            for group in ("head", "mid", "tail")
        )
    )
    print(f"Saved popularity table to {DEFAULT_OUTPUT}")
    print(f"Saved popularity plot to {DEFAULT_PLOT}")


if __name__ == "__main__":
    main()
