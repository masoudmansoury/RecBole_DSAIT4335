"""Export every input Task 2.3/2.4 needs, in the Task 2.1 evaluator's JSON format.

Run from the repository root:
    python code/Part2/Task2.3/export_data.py

Outputs:
    saved/splits/ml-100k.{train,valid,test}.tsv              split (git-ignored)
    run_outputs/Part2/Task2.3/recommendations/*.json         test top-k lists
"""

import glob
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path, PureWindowsPath

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "code/Part1/Part1.5"))

# Part 1.5 puts Part 1.4 on sys.path, so its utils module is importable afterwards.
from predict import SavedHybrid  # noqa: E402
from run_hybrid import MODEL_DIR, Experiment, latest_checkpoints  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import get_model, init_seed  # noqa: E402
from utils import DEVICE, get_scores, top_items  # noqa: E402

OUT = ROOT / "run_outputs/Part2/Task2.3"
SPLIT_DIR = ROOT / "saved/splits"
HYBRID_DIR = ROOT / "run_outputs/Part1/Part1.5/best_models"
BASELINE_DIR = ROOT / "run_outputs/Part1/Part1.1"

HYBRIDS = ["Cascade", "FeatureCombination"]
# Components of the retained hybrids. LightGCN needs a deep list for the
# Cascade candidate analysis; the evaluator truncates to its own k.
COMPONENTS = {"LightGCN": 100, "SLIMElastic": 100}
BASELINES = {"Pop": 100, "Random": 100}


class LocalSavedHybrid(SavedHybrid):
    """SavedHybrid that accepts base checkpoints saved on another machine.

    Part 1.5 stores absolute (Windows) paths, so its exact path check always
    fails elsewhere. Comparing file names keeps the same safety guarantee.
    """

    def __init__(self, checkpoint_path, experiment):
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if self.checkpoint.get("format_version") != 2:
            raise ValueError("Use a checkpoint saved by the rebuilt Part 1.5")
        saved = {name: PureWindowsPath(path).name
                 for name, path in self.checkpoint["base_checkpoints"].items()}
        local = {Path(path).parent.name: Path(path).name
                 for path in latest_checkpoints(str(MODEL_DIR))}
        if saved != local:
            raise ValueError(f"Base checkpoints differ: saved={saved}, local={local}")
        self.experiment = experiment


def write_recommendations(name, rec_by_user, config, k, out_dir):
    payload = {
        "dataset": config["dataset"],
        "model": name,
        "split": "test",
        "k": k,
        "seed": config["seed"],
        "eval_args": config["eval_args"],
        "recommendations": {user: {"items": items} for user, items in rec_by_user.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config['dataset']}_{name}_test_top{k}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {name}: {len(rec_by_user)} users, top-{k} -> {path.relative_to(ROOT)}")


def to_tokens(rec, data):
    dataset = data.dataset
    users = dataset.id2token(dataset.uid_field, data.test.users.cpu().numpy())
    items = dataset.id2token(dataset.iid_field, rec.cpu().numpy())
    return {str(user): items[i].tolist() for i, user in enumerate(users)}


def export_split():
    # Reuse the team's split exporter; its output was verified to reproduce
    # the Part 1.4 hybrid metrics exactly.
    subprocess.run([
        sys.executable, str(ROOT / "save_split.py"), "--model", "BPR", "--dataset", "ml-100k",
        "--config_files", "recbole/config/BPR/ml-100k.yaml", "--output_dir", str(SPLIT_DIR),
    ], check=True, cwd=ROOT)


def export_hybrids(experiment, rec_dir):
    for name in HYBRIDS:
        hybrid = LocalSavedHybrid(HYBRID_DIR / f"{name}.pth", experiment)
        write_recommendations(name, hybrid.recommend(split="test"),
                              experiment.data.config, 10, rec_dir)

    # FeatureCombination ablation: identical pipeline with the genre weight set to 0.
    hybrid = LocalSavedHybrid(HYBRID_DIR / "FeatureCombination.pth", experiment)
    hybrid.checkpoint = {**hybrid.checkpoint, "fitted_state": {"best_weight": 0.0}}
    write_recommendations("FeatureCombination_control", hybrid.recommend(split="test"),
                          experiment.data.config, 10, rec_dir)


def export_components(experiment, rec_dir):
    for name, k in COMPONENTS.items():
        scores = experiment.scores[id(experiment.models[name])]
        rec = top_items(scores, experiment.data.test, k)
        write_recommendations(name, to_tokens(rec, experiment.data), experiment.data.config, k, rec_dir)


def export_baselines(experiment, rec_dir):
    data = experiment.data
    for name, k in BASELINES.items():
        path = max(glob.glob(str(BASELINE_DIR / name / "*.pth")), key=os.path.getmtime)
        checkpoint = torch.load(path, weights_only=False, map_location=DEVICE)
        config = checkpoint["config"]
        config["data_path"] = str(ROOT / "dataset" / config["dataset"])
        config["device"] = DEVICE
        config["use_gpu"] = DEVICE.type == "cuda"
        config["show_progress"] = False
        if (config["seed"], config["eval_args"]) != (data.config["seed"], data.config["eval_args"]):
            raise ValueError(f"{name} was trained under a different protocol")
        init_seed(config["seed"], config["reproducibility"])
        train, _, _ = data_preparation(config, create_dataset(config))
        model = get_model(config["model"])(config, train._dataset).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
        init_seed(config["seed"], config["reproducibility"])
        rec = top_items(get_scores(model, data), data.test, k)
        write_recommendations(name, to_tokens(rec, data), data.config, k, rec_dir)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    rec_dir = OUT / "recommendations"
    export_split()
    print("loading base models and scores ...")
    with redirect_stdout(io.StringIO()):
        experiment = Experiment()
    export_hybrids(experiment, rec_dir)
    export_components(experiment, rec_dir)
    export_baselines(experiment, rec_dir)


if __name__ == "__main__":
    main()
