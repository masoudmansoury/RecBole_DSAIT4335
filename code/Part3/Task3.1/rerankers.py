"""Score-aware rerankers for Task 3.1."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean


Candidate = tuple[str, float]


def _validate_lambda(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("lambda must be between 0 and 1")


def _normalised_relevance(candidates: Sequence[Candidate]) -> dict[str, float]:
    values = [score for _, score in candidates]
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return {item: 1.0 for item, _ in candidates}
    return {item: (score - low) / (high - low) for item, score in candidates}


def _intra_list_diversity(
    items: Sequence[str],
    genres: Mapping[str, Iterable[str]],
) -> float:
    """Mean pairwise genre Jaccard distance, as used by the evaluator."""

    known = [item for item in items if set(genres.get(item, ()))]
    if len(known) < 2:
        return 0.0
    distances = []
    for index, left in enumerate(known):
        left_genres = set(genres[left])
        for right in known[index + 1 :]:
            right_genres = set(genres[right])
            similarity = len(left_genres & right_genres) / len(
                left_genres | right_genres
            )
            distances.append(1.0 - similarity)
    return fmean(distances)


def _validate_candidates(candidates: Sequence[Candidate], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(candidates) < k:
        raise ValueError(f"received {len(candidates)} candidates for top-{k}")
    items = [item for item, _ in candidates]
    if len(items) != len(set(items)):
        raise ValueError("candidate items must be unique")
    if any(not math.isfinite(score) for _, score in candidates):
        raise ValueError("candidate scores must be finite")


def mmr_rerank(
    candidates: Sequence[Candidate],
    genres: Mapping[str, Iterable[str]],
    k: int = 10,
    lambda_: float = 0.2,
) -> list[Candidate]:
    """Greedily maximise ``(1-lambda)*rel(L) + lambda*ILD(L)``.

    Lambda is the diversity weight, so
    ``lambda=0`` reproduces the original score ranking.
    """

    _validate_candidates(candidates, k)
    _validate_lambda(lambda_)
    relevance = _normalised_relevance(candidates)
    original_rank = {item: rank for rank, (item, _) in enumerate(candidates)}
    remaining = dict(candidates)
    selected: list[Candidate] = []

    while len(selected) < k:
        selected_items = [item for item, _ in selected]
        selected_relevance = sum(relevance[item] for item in selected_items)

        def objective(item: str) -> tuple[float, int]:
            new_list = selected_items + [item]
            list_relevance = selected_relevance + relevance[item]
            value = (
                (1.0 - lambda_) * list_relevance
                + lambda_ * _intra_list_diversity(new_list, genres)
            )
            return value, -original_rank[item]

        chosen = max(remaining, key=objective)
        selected.append((chosen, remaining.pop(chosen)))

    return selected


def _genre_distribution(
    items: Sequence[str],
    genres: Mapping[str, Iterable[str]],
    rank_weighted: bool = False,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for rank, item in enumerate(items, start=1):
        item_genres = set(genres.get(item, ()))
        if not item_genres:
            continue
        weight = 1.0 / math.log2(rank + 1) if rank_weighted else 1.0
        contribution = weight / len(item_genres)
        for genre in item_genres:
            counts[genre] += contribution
    total = sum(counts.values())
    return {genre: value / total for genre, value in counts.items()} if total else {}


def _calibration_kl(
    profile: Mapping[str, float],
    items: Sequence[str],
    genres: Mapping[str, Iterable[str]],
    alpha: float,
) -> float:
    recommended = _genre_distribution(items, genres, rank_weighted=True)
    return sum(
        probability
        * math.log2(
            probability
            / ((1.0 - alpha) * recommended.get(genre, 0.0) + alpha * probability)
        )
        for genre, probability in profile.items()
    )


def calibration_rerank(
    candidates: Sequence[Candidate],
    history: Sequence[str],
    genres: Mapping[str, Iterable[str]],
    k: int = 10,
    lambda_: float = 0.2,
    alpha: float = 0.01,
) -> list[Candidate]:
    """Greedily maximise ``(1-lambda)*rel(L) - lambda*KL(p,q(L))``.

    Lambda is the calibration weight, so
    ``lambda=0`` reproduces the original score ranking. The KL term compares
    the training-history genre profile with the rank-weighted genre
    distribution of the partial reranked list.
    """

    _validate_candidates(candidates, k)
    _validate_lambda(lambda_)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    profile = _genre_distribution(history, genres)
    if not profile or math.isclose(lambda_, 0.0):
        return list(candidates[:k])

    relevance = _normalised_relevance(candidates)
    original_rank = {item: rank for rank, (item, _) in enumerate(candidates)}
    remaining = dict(candidates)
    selected: list[Candidate] = []

    while len(selected) < k:
        selected_items = [item for item, _ in selected]
        selected_relevance = sum(relevance[item] for item in selected_items)

        def objective(item: str) -> tuple[float, int]:
            new_list = selected_items + [item]
            miscalibration = _calibration_kl(
                profile, new_list, genres, alpha
            )
            value = (
                (1.0 - lambda_) * (selected_relevance + relevance[item])
                - lambda_ * miscalibration
            )
            return value, -original_rank[item]

        chosen = max(remaining, key=objective)
        selected.append((chosen, remaining.pop(chosen)))

    return selected
