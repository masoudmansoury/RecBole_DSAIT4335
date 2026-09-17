"""
save_recommendations.py

Loads a trained RecBole checkpoint, re-creates the exact train/valid/test
split that was used to train it, computes the top-K recommendation list
for every user in the test set with full_sort_topk, and writes the result
to a JSON file that is independent of RecBole internals (external item
tokens, plain floats) so it can be re-used later -- e.g. by a different
script that computes a new metric, or to compare several models offline.

Usage:
    python save_recommendations.py \
        --model_file saved/BPR-<timestamp>.pth \
        --k 10 \
        --output_dir saved/recommendations
"""
import argparse
import json
import os

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,
                         help="path to a .pth checkpoint produced by run_recbole.py")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="saved/recommendations")
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.model_file
    )
    model.eval()

    # every user that appears in the test set
    uid_field = dataset.uid_field
    test_user_ids = test_data.dataset.inter_feat[uid_field].numpy()
    uid_series = sorted(set(int(u) for u in test_user_ids))

    topk_score, topk_iid_list = full_sort_topk(
        uid_series, model, test_data, k=args.k, device=config["device"]
    )

    external_users = dataset.id2token(uid_field, uid_series)
    external_items = dataset.id2token(
        dataset.iid_field, topk_iid_list.cpu().numpy()
    )

    results = {}
    for row, user_token in enumerate(external_users):
        results[str(user_token)] = {
            "items": external_items[row].tolist(),
            "scores": topk_score[row].cpu().tolist(),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f'{config["dataset"]}_{config["model"]}_top{args.k}.json'
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "k": args.k,
                "seed": config["seed"],
                "eval_args": config["eval_args"],
                "recommendations": results,
            },
            f,
            indent=2,
        )
    print(f"Saved top-{args.k} recommendations for {len(results)} users -> {out_path}")


if __name__ == "__main__":
    main()
