"""Step 1 of Task 2.3: evaluate every exported list with the Task 2.1 evaluator.

Run from the repository root (after export_data.py):
    python code/Part2/Task2.3/run_eval.py

For each file in run_outputs/Part2/Task2.3/recommendations/ this calls
code/Part2/Part2.1/evaluate.py unchanged and keeps its outputs:
    run_outputs/Part2/Task2.3/eval/<model>.json   aggregate metrics
    run_outputs/Part2/Task2.3/eval/<model>.tsv    per-user metrics
It then collects the aggregates into one comparison table:
    run_outputs/Part2/Task2.3/eval/summary.csv
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT / "code/Part2/Part2.1/evaluate.py"
SPLITS = ROOT / "saved/splits"
REC_DIR = ROOT / "run_outputs/Part2/Task2.3/recommendations"
OUT = ROOT / "run_outputs/Part2/Task2.3/eval"
K = 10

# Role of every exported list, and which retained hybrid(s) it belongs to.
MODELS = {
    "Cascade": ("hybrid", "Cascade"),
    "LightGCN": ("component", "Cascade"),
    "SLIMElastic": ("component", "Cascade"),
    "FeatureCombination": ("hybrid", "FeatureCombination"),
    "FeatureCombination_control": ("ablation", "FeatureCombination"),
    "Pop": ("baseline", "-"),
    "Random": ("baseline", "-"),
}
ACCURACY = ["precision", "recall", "f1", "hit", "mrr", "ndcg", "map"]
BEYOND = ["ild", "novelty", "miscalibration", "popularity_deviation",
          "catalog_coverage", "exposure_gini", "exposure_entropy"]


def recommendation_file(model):
    matches = sorted(REC_DIR.glob(f"ml-100k_{model}_test_top*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one recommendation file for {model}, found {matches}")
    return matches[0]


def evaluate(recommendations, out_dir, name):
    """Run the Task 2.1 evaluator unchanged on one recommendation file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(EVALUATOR),
        "--train", str(SPLITS / "ml-100k.train.tsv"),
        "--valid", str(SPLITS / "ml-100k.valid.tsv"),
        "--test", str(SPLITS / "ml-100k.test.tsv"),
        "--recommendations", str(recommendations),
        "--k", str(K),
        "--output", str(out_dir / f"{name}.json"),
        "--per-user-output", str(out_dir / f"{name}.tsv"),
    ], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))


def metric_columns(result):
    row = {f"{name}@{K}": result["accuracy"][name] for name in ACCURACY}
    row.update({name: result["beyond_accuracy"][name] for name in BEYOND})
    for attribute in ("gender", "age"):
        gaps = result["group_fairness"][attribute]["pairwise_gru"].values()
        row[f"ndcg_gap_{attribute}_max"] = max(gaps)
    return row


def write_table(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 4) if isinstance(value, float) else value
                             for key, value in row.items()})
    print(f"saved {path.relative_to(ROOT)}")


def main():
    rows = []
    for model, (role, hybrid) in MODELS.items():
        result = evaluate(recommendation_file(model), OUT, model)
        rows.append({"model": model, "role": role, "hybrid": hybrid,
                     "users": result["evaluated_users"], **metric_columns(result)})
        print(f"evaluated {model}")
    write_table(rows, OUT / "summary.csv")


if __name__ == "__main__":
    main()
