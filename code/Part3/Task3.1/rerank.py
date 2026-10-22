"""Apply the Task 3.1 diversification or calibration reranker."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rerankers import Candidate, calibration_rerank, mmr_rerank


def _field_name(value: str) -> str:
    return value.split(":", 1)[0]


def load_genres(path: Path) -> dict[str, frozenset[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        reader.fieldnames = [_field_name(field) for field in reader.fieldnames]
        if "item_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain item_id")
        genre_field = "class" if "class" in reader.fieldnames else "genres"
        if genre_field not in reader.fieldnames:
            raise ValueError(f"{path} must contain class or genres")
        return {
            row["item_id"]: frozenset(
                value for value in row[genre_field].replace("|", " ").split() if value
            )
            for row in reader
        }


def load_histories(path: Path) -> dict[str, list[str]]:
    histories: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        reader.fieldnames = [_field_name(field) for field in reader.fieldnames]
        if not {"user_id", "item_id"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain user_id and item_id")
        for row in reader:
            histories[row["user_id"]].append(row["item_id"])
    return dict(histories)


def load_scored_candidates(
    path: Path, candidate_k: int
) -> tuple[dict[str, object], dict[str, list[Candidate]]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = payload.get("recommendations") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        raise ValueError(f"{path} must contain a recommendations object")

    result: dict[str, list[Candidate]] = {}
    for user_id, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid recommendation row for user {user_id}")
        items, scores = row.get("items"), row.get("scores")
        if not isinstance(items, list) or not isinstance(scores, list):
            raise ValueError(
                f"user {user_id} must have parallel items and scores lists"
            )
        if len(items) != len(scores):
            raise ValueError(f"items/scores length mismatch for user {user_id}")
        if len(items) < candidate_k:
            raise ValueError(
                f"user {user_id} has {len(items)} candidates; expected {candidate_k}"
            )
        candidates = [
            (str(item), float(score))
            for item, score in zip(items[:candidate_k], scores[:candidate_k])
        ]
        if len({item for item, _ in candidates}) != len(candidates):
            raise ValueError(f"duplicate candidates for user {user_id}")
        result[str(user_id)] = candidates

    metadata = {key: value for key, value in payload.items() if key != "recommendations"}
    return metadata, result


def rerank_payload(
    metadata: dict[str, object],
    candidates_by_user: dict[str, list[Candidate]],
    genres: dict[str, frozenset[str]],
    method: str,
    k: int,
    lambda_: float,
    histories: dict[str, list[str]] | None = None,
    alpha: float = 0.01,
) -> dict[str, object]:
    output_rows = {}
    for user_id, candidates in candidates_by_user.items():
        if method == "diversity":
            selected = mmr_rerank(candidates, genres, k, lambda_)
        elif method == "calibration":
            if histories is None:
                raise ValueError("calibration reranking requires training histories")
            selected = calibration_rerank(
                candidates,
                histories.get(user_id, []),
                genres,
                k,
                lambda_,
                alpha,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        output_rows[user_id] = {
            "items": [item for item, _ in selected],
            "scores": [score for _, score in selected],
        }

    source_model = str(metadata.get("model", "unknown"))
    return {
        **metadata,
        "model": f"{source_model}+{method}",
        "source_model": source_model,
        "k": k,
        "reranker": {
            "method": method,
            "lambda": lambda_,
            "calibration_alpha": alpha if method == "calibration" else None,
        },
        "recommendations": output_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("diversity", "calibration"), required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--lambda", dest="lambda_", type=float, default=0.2,
        help="beyond-accuracy weight from the course objective; 0 keeps the score ranking",
    )
    parser.add_argument("--calibration-alpha", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.method == "calibration" and args.train is None:
        parser.error("--train is required for calibration reranking")
    if args.candidate_k < args.k:
        parser.error("--candidate-k must be at least --k")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata, candidates = load_scored_candidates(args.input, args.candidate_k)
    genres = load_genres(args.items)
    histories = load_histories(args.train) if args.train else None
    output = rerank_payload(
        metadata,
        candidates,
        genres,
        args.method,
        args.k,
        args.lambda_,
        histories,
        args.calibration_alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved {len(candidates)} users -> {args.output}")


if __name__ == "__main__":
    main()
