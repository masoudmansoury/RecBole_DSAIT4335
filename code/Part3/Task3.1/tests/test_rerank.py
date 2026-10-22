import json
import sys
import tempfile
import unittest
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from rerank import load_scored_candidates, main, rerank_payload
from rerankers import calibration_rerank, mmr_rerank


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        self.genres = {"a": {"Action"}, "b": {"Action"}, "c": {"Comedy"}}

    def test_mmr_trades_relevance_for_diversity(self):
        result = mmr_rerank(
            self.candidates, self.genres, k=2, lambda_=0.5
        )
        self.assertEqual([item for item, _ in result], ["a", "c"])

    def test_relevance_only_preserves_ranking(self):
        result = mmr_rerank(
            self.candidates, self.genres, k=3, lambda_=0.0
        )
        self.assertEqual(result, self.candidates)

    def test_calibration_uses_user_profile(self):
        result = calibration_rerank(
            self.candidates,
            history=["a", "c"],
            genres=self.genres,
            k=2,
            lambda_=0.8,
        )
        self.assertEqual([item for item, _ in result], ["a", "c"])

    def test_missing_profile_falls_back_to_relevance(self):
        result = calibration_rerank(
            self.candidates,
            history=[],
            genres=self.genres,
            k=2,
            lambda_=0.8,
        )
        self.assertEqual(result, self.candidates[:2])


class InputContractTests(unittest.TestCase):
    def test_scores_are_required(self):
        with tempfile.TemporaryDirectory(dir=TASK_DIR / "tests") as directory_name:
            path = Path(directory_name) / "input.json"
            path.write_text(
                json.dumps(
                    {"recommendations": {"u1": {"items": ["a", "b"]}}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "items and scores"):
                load_scored_candidates(path, candidate_k=2)

    def test_output_remains_evaluator_compatible(self):
        payload = rerank_payload(
            {"dataset": "ml-100k", "model": "Example", "split": "test", "k": 3},
            {"u1": [("a", 1.0), ("b", 0.9), ("c", 0.8)]},
            {"a": frozenset({"Action"}), "b": frozenset({"Action"}),
             "c": frozenset({"Comedy"})},
            method="diversity",
            k=2,
            lambda_=0.5,
        )
        self.assertEqual(payload["k"], 2)
        self.assertEqual(payload["source_model"], "Example")
        self.assertEqual(payload["recommendations"]["u1"]["items"], ["a", "c"])
        self.assertEqual(payload["recommendations"]["u1"]["scores"], [1.0, 0.8])

    def test_calibration_cli(self):
        with tempfile.TemporaryDirectory(dir=TASK_DIR / "tests") as directory_name:
            directory = Path(directory_name)
            candidates = directory / "candidates.json"
            items = directory / "items.item"
            train = directory / "train.tsv"
            output = directory / "output.json"
            candidates.write_text(
                json.dumps(
                    {
                        "dataset": "ml-100k",
                        "model": "Example",
                        "split": "test",
                        "k": 3,
                        "recommendations": {
                            "u1": {
                                "items": ["a", "b", "c"],
                                "scores": [1.0, 0.9, 0.8],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            items.write_text(
                "item_id:token\tclass:token_seq\n"
                "a\tAction\n"
                "b\tAction\n"
                "c\tComedy\n",
                encoding="utf-8",
            )
            train.write_text(
                "user_id\titem_id\nu1\ta\nu1\tc\n", encoding="utf-8"
            )
            main(
                [
                    "--input", str(candidates),
                    "--items", str(items),
                    "--train", str(train),
                    "--method", "calibration",
                    "--candidate-k", "3",
                    "--k", "2",
                    "--lambda", "0.8",
                    "--output", str(output),
                ]
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendations"]["u1"]["items"], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
