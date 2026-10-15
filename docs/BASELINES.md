# Reproducing the comparison methods

This release includes the **15 external baseline adaptations used in OA-CXR**:
eight regressors, three visual ranking signals and four report confidence
signals. These are our adaptations to anatomical retention, not claims to
reproduce every experiment in the original baseline papers. Internal OA
revisions, ablations, Global DenseNet and mask-geometry controls are not counted
as additional external baselines.

| Methods | Input and fitting | Published implementation |
|---|---|---|
| HGB, XGBoost, CatBoost, Ridge | Same 1,253 OA features; fit split only; fixed hyperparameters | `baselines/tabular.py`, per-method `config.json` |
| TabM | Same OA features; official TabM 0.0.3; 30 epochs | `baselines/foundation_regressors.py` |
| BiomedCLIP, CheXzero, DINOv2 + MLP | Frozen official encoder; full-frame 224-pixel input; new residual head, 30 epochs | `baselines/foundation_encoders.py`, `foundation_regressors.py` |
| CheXpert-only | Disease-absence score: 1 − sigmoid(logit); ranking only | `baselines/chexpert/run.py` |
| AutoQC CXR-NIQE | Negative image-quality distance; ranking only | `baselines/autoqc/run.py` |
| BioViL-T | Negative minus positive prompt similarity; ranking only | `baselines/biovil_t/run.py` |
| Report log probability, sentence log probability, negative token entropy | Same frozen MAIRA-2 reports and teacher-forced token statistics | `baselines/reporting.py`, three method adapters |
| SCUQ-RadGraph | Ten stochastic reports per eligible input; sentence-level official SCUQ adaptation | `baselines/reporting_recovery.py`, `baselines/scuq_radgraph/run.py` |

The machine-readable list is [registry.json](../baselines/registry.json).
The original fixed configs retain historical planning/status fields; they are
not current execution receipts. A newly executed run writes its own status.

## Environments and upstream assets

Use separate environments because the frozen-encoder and report toolchains
use different Transformers/timm versions. Start with the main README's Torch
installation. For the eight regression methods:

```bash
python -m pip install -e '.[training,download,dev,baseline-tabular,baseline-neural,baseline-encoders]'
python -m pip check
```

The fixed encoder stack uses open_clip 2.23.0, timm 0.9.12 and Transformers
4.35.2. MAIRA-2 uses the existing `requirements/rebuild-linux.lock.txt` in a
separate environment. BioViL-T and RadGraph use
`requirements/rebuild-baselines.lock.txt`; AutoQC uses
`requirements/rebuild-autoqc.lock.txt`. These Linux/Python 3.10 locks include
CUDA 12.4 Torch builds where applicable. Install the matching Torch index first.
The lightweight tabular-only lock is `requirements/rebuild-tabular.lock.txt`.

Fetch assets directly from their upstream authors into a fresh, ignored local
directory. The downloader fixes repository revisions and verifies file SHA256:

```bash
python scripts/public/baseline_assets.py --method biomedclip --output asset_staging/biomedclip
python scripts/public/baseline_assets.py --method chexzero --output asset_staging/chexzero
python scripts/public/baseline_assets.py --method dinov2 --output asset_staging/dinov2
python scripts/public/baseline_assets.py --method chexpert --output asset_staging/chexpert
python scripts/public/baseline_assets.py --method autoqc --output asset_staging/autoqc
python scripts/public/baseline_assets.py --method biovil_t --output asset_staging/biovil_t
```

Run each command in its appropriate environment. TabM is trained from scratch.
CheXzero uses the fixed single official checkpoint, not an ensemble. The
CheXpert conversion accepts only the exact pinned official legacy file before
converting it to a tensor-only state. Different serialization bytes after this
verified conversion are recorded in the generated local config.

Third-party implementations and pretrained weights are **not bundled** in Git
or mirrored on our model page. Upstream access terms still apply, especially
to MAIRA-2. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Regression: features, fitting and prediction

First prepare the public caches and download the OA paper model as described
in the README. Using the published OA image checkpoint ensures that the
same-feature comparisons use the paper representation. Features from a newly
trained OA image model define a new experiment and must be used consistently
for every split and downstream learner.

```bash
for split in fit dev test; do
  python scripts/public/baseline_features.py export --method oa \
    --weights weights/main --cache data/main/cache --split "$split" \
    --gpu-uuid GPU-... --output runs/features/oa/"$split"
  python scripts/public/baseline_features.py attach-labels \
    --features runs/features/oa/"$split" --cache data/main/cache --split "$split" \
    --output runs/labeled/oa/"$split"
done

for method in hgb xgboost catboost ridge tabm; do
  python scripts/public/train_baseline.py --method "$method" \
    --fit runs/labeled/oa/fit/manifest.json --dev runs/labeled/oa/dev/manifest.json \
    --output runs/baselines/"$method"/train
  python scripts/public/predict_baseline.py \
    --training runs/baselines/"$method"/train --manifest runs/features/oa/test/manifest.json \
    --output runs/baselines/"$method"/test
done
```

HGB/XGBoost/CatBoost/Ridge ignore the development argument and fit on the fit
split only. TabM and the three encoder heads use full-development MAE to select
the earliest best epoch. They require the complete 19,086/2,609 fit/dev source
counts and 13 presentations × three queries per source. Each neural method
performs 30 epochs and 10,920 optimizer steps. TabM uses effective batch 2,048,
microbatch 256, width 512, three blocks, ensemble size 32 and dropout 0.1.
The three encoder heads use the same width-256, two-block OA residual readout.
An optional `--gpu-uuid` enables neural fitting on an explicitly selected idle GPU.

For each of `biomedclip`, `chexzero` and `dinov2`, export all splits with that
encoder, attach the same independent targets, and train/predict its head:

```bash
method=dinov2
for split in fit dev test; do
  python scripts/public/baseline_features.py export --method "$method" \
    --assets asset_staging/"$method"/asset_manifest.json \
    --cache data/main/cache --split "$split" --gpu-uuid GPU-... \
    --output runs/features/"$method"/"$split"
  python scripts/public/baseline_features.py attach-labels \
    --features runs/features/"$method"/"$split" --cache data/main/cache --split "$split" \
    --output runs/labeled/"$method"/"$split"
done
python scripts/public/train_baseline.py --method "$method" \
  --fit runs/labeled/"$method"/fit/manifest.json --dev runs/labeled/"$method"/dev/manifest.json \
  --output runs/baselines/"$method"/train
python scripts/public/predict_baseline.py --training runs/baselines/"$method"/train \
  --manifest runs/features/"$method"/test/manifest.json --output runs/baselines/"$method"/test
```

Repeat feature export and prediction for `external_nih` and `external_shenzhen`
using their prepared caches. Keep the same model and precision across splits.
GPU OA export uses batch 64; independent encoders use batch 128 and BF16.
Omit the GPU argument only for CPU technical checks; FP32 CPU values need not
equal BF16 reference values exactly. Prediction does not load target arrays.

## Visual ranking methods

Materialize the exact cache pixels once; all visual methods receive these PNGs:

```bash
python scripts/public/baseline_inputs.py --cache data/main/cache --split test --output runs/presented/test
python -m baselines.chexpert.run --inputs runs/presented/test/inputs.jsonl \
  --expected-inputs 36296 --config asset_staging/chexpert/config.json \
  --device cuda:0 --gpu-uuid GPU-... --output runs/baselines/chexpert/test
```

Use `baselines.biovil_t.run` with its local config in the BioViL-T environment.
Use `baselines.autoqc.run --device cpu` with its local config in the AutoQC
environment. BioViL-T retains its official resize-512/center-crop-448 transform;
AutoQC applies its original internal NIQE crop. These methods are ranking
signals and must not be reported as calibrated retention regressors.

Evaluate any chosen common panel with paired source/family bootstrap:

```bash
python scripts/public/evaluate.py --weights weights/main --cache data/main/cache \
  --split test --gpu-uuid GPU-... --output runs/oa/test
python scripts/public/evaluation_scores.py --cache data/main/cache --split test \
  --evaluation runs/oa/test --output runs/oa/test_scores
python scripts/public/compare_baselines.py --cache data/main/cache --split test \
  --prediction oa_cxr=runs/oa/test_scores/predictions.jsonl \
  --prediction hgb=runs/baselines/hgb/test/predictions.jsonl \
  --prediction chexpert=runs/baselines/chexpert/test/scores.jsonl \
  --reference oa_cxr --bootstrap 1000 --output runs/comparison/test
```

Add one `--prediction` for every method in the intended panel. Missing rows
are errors; explicit failed scores remain in the nominal denominator. Metrics
are computed on the common scored queries, with individual success counts.

## Report confidence methods

`baseline_inputs.py` also writes `report_inputs.jsonl`: the first 128 sources
per dataset under the fixed source-hash rule, chosen before generation. The
main test split supplies 3,328 reports; each external source supplies 1,664.
Run separately for each prepared cache and use the actual recorded input count.

After obtaining authorized access to MAIRA-2, fetch the fixed upstream assets:

```bash
python scripts/public/baseline_assets.py --method maira_logprob --output asset_staging/maira
python scripts/public/baseline_assets.py --method scuq_radgraph --output asset_staging/scuq
python -m baselines.reporting generate --inputs runs/presented/test/report_inputs.jsonl \
  --expected-inputs 3328 --config asset_staging/maira/generation_config.json \
  --gpu-uuid GPU-... --output runs/reports/test/greedy
python -m baselines.reporting tokens --generation runs/reports/test/greedy \
  --config asset_staging/maira/generation_config.json --gpu-uuid GPU-... --output runs/reports/test/tokens
for method in maira_logprob sentence_logprob token_entropy; do
  python -m baselines."$method".run --token-run runs/reports/test/tokens --output runs/reports/test/"$method"
done
python -m baselines.reporting_recovery --generation runs/reports/test/greedy \
  --config asset_staging/maira/generation_config.json --gpu-uuid GPU-... --output runs/reports/test/samples
python -m baselines.scuq_radgraph.run --generation runs/reports/test/greedy \
  --samples runs/reports/test/samples --config asset_staging/scuq/config.json \
  --gpu-uuid GPU-... --output runs/reports/test/scuq_radgraph
```

The last command runs in the RadGraph environment. The parser-safe sampling
entrypoint preserves raw outcomes and does not replace failed/truncated samples.
Historical recovery of old failed attempts is optional and not needed for a
fresh run. Greedy generation is BF16/batch 16/max 512 tokens; SCUQ uses ten
singleton draws, temperature 1, top-p 1 and top-k 0. Scores refer to the same
eligible negative claim occurrences, not invented absent statements.

```bash
python scripts/public/compare_report_baselines.py --cache data/main/cache --split test \
  --generation runs/reports/test/greedy --image-score oa_cxr=runs/oa/test_scores/predictions.jsonl \
  --report-score maira_logprob=runs/reports/test/maira_logprob/scores.jsonl \
  --report-score sentence_logprob=runs/reports/test/sentence_logprob/scores.jsonl \
  --report-score token_entropy=runs/reports/test/token_entropy/scores.jsonl \
  --report-score scuq_radgraph=runs/reports/test/scuq_radgraph/scores.jsonl \
  --reference oa_cxr --bootstrap 1000 --output runs/reports/test/comparison
```

Add image-score arguments for the image baselines in the chosen panel. Retain
all reports, including those with zero eligible claims, in the resampling
population. The historical broad panel has 741 common claims; the separate
encoder/TabM panel has 784. Those panels must not be merged into one ranking,
and a fresh stochastic SCUQ run need not reproduce exactly the same failures.

## Evidence and scope

[Public reference metrics](../reproducibility/baseline_reference_metrics.json)
contain aggregate image results. [Training receipts](../reproducibility/baseline_training_reference.json)
record completed historical fitting for all eight regressors, with source
artifact hashes. [Release validation](../reproducibility/baseline_release_validation.json)
distinguishes checks run during publication from historical complete experiments.
Publishing code and passing synthetic checks do not establish that all large
training, upstream downloads and report-generation jobs were rerun from scratch.

The private clinical cohort is documented separately in
[PRIVATE_DATA.md](PRIVATE_DATA.md); it is not part of the downloadable benchmarks.
