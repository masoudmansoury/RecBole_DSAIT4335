# Task 3.1

Input recommendations json must provide top-100 candidates (or top-k by setting the arg `--candidate-k` when running `rerank.py`) and model scores in the existing
recommendation JSON format:

```json
"recommendations": {
  "196": {"items": ["275", "204"], "scores": [8.4, 7.9]}
}
```

Run diversification:

```powershell
python code/Part3/Task3.1/rerank.py `
  --input path/to/scored_top100.json `
  --items dataset/ml-100k/ml-100k.item `
  --method diversity `
  --lambda 0.2 `
  --output run_outputs/Part3/Task3.1/diversity.json
```

Run calibration:

```powershell
python code/Part3/Task3.1/rerank.py `
  --input path/to/scored_top100.json `
  --items dataset/ml-100k/ml-100k.item `
  --train saved/splits/ml-100k.train.tsv `
  --method calibration `
  --lambda 0.2 `
  --output run_outputs/Part3/Task3.1/calibration.json
```

Use the same candidate file and candidate-pool size for every reranker. The
parameter follows the course notation: `lambda` is the diversity/calibration
weight, and `lambda=0` is the no-reranking control.
