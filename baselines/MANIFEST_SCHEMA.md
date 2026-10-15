# Feature manifest

`scripts/public/baseline_features.py` writes `oa-cxr-tabular-dataset-v1` manifests
with target `anatomical_retention`, ordered `feature_names`, a shared
`feature_protocol_sha256`, and SHA-bound `samples`/`features` paths.

The NPZ contains float32 `X`, Unicode `row_ids`, boolean `available`, and optionally
float32 `y` in [0,1] after the separate label-attachment action. Prediction must
not load `y`. Feature extraction never reads annotation masks or targets.

Each JSONL sample identifies `row_id`, `input_id`, `source_id`, `split_group_id`,
`dataset`, `split`, `finding`, and `source_image_sha256`. Identity/order must
match the NPZ exactly. All variants of a source/group stay in one partition.
The three findings are `pleural_effusion`, `pneumothorax`, `consolidation`.

Paths may be relative to the manifest. The same model, precision and feature
protocol must be used across fit, dev and evaluation. A shape match alone does
not establish feature compatibility. See [commands](../docs/BASELINES.md).
