# Reproducing OA-CXR

## Released scientific recipe

| Stage | Fixed recipe |
|---|---|
| Input | 256×256 grayscale, aspect-preserving letterbox; original normalization |
| Initialization | Official XRV CheXpert DenseNet121, SHA-pinned conversion |
| Image model | Trainable encoder, auxiliary two-lung decoder, finding-conditioned retention heads |
| Image fitting | seed 17; 13 epochs; physical batch 256; AdamW lr 1e-4 / wd 1e-4; BF16; gradient L2 clip 5 |
| Image supervision | SmoothL1 beta 0.1 + 0.25 × (segmentation BCE + 0.5 × Dice loss) |
| Source sampling | One cyclic variant per source per epoch; all 13 variants over 13 epochs |
| Image selection | MAE on the predetermined 512-source dev subset; earliest epoch wins ties |
| Feature export | Frozen image checkpoint, BF16 image forward, FP32 stored features, batch 64 |
| Readout input | 1250 model features + three finding one-hot columns |
| Readout | Fit-only standardization; width 256, two residual blocks, no dropout |
| Readout fitting | Three candidates, each 30 epochs, batch 2048, CPU FP32, 12 threads; AdamW lr 1e-3 / wd 1e-4; cosine decay to 1e-5 |
| Candidates | Linear output + L1; linear output + SmoothL1 beta .02; original-logit residual + L1 |
| Final selection | All 2,609 dev source images; lexicographic (MAE, selected epoch, candidate order) |
| Inference | Selected linear readout is clipped to [0,1] only for scoring |

The image stage performs 975 optimizer updates. Each readout candidate performs
10,920 updates. Fit consists of 19,086 sources and 744,354 regional queries in
the readout stage. Calibration/test/external arrays are not opened by training.

The orchestrator is `scripts/public/train_pipeline.py`; algorithm implementations
are in `scripts/rebuild/train.py`, `export.py`, `train_neural_readout.py`,
`src/oa_cxr/rebuild/vision.py` and `neural_readout.py`. The public adaptation
replaces dependence on the author's historical visual-inspection receipts with
explicit automated integrity receipts for newly prepared caches. It does not
claim an automated integrity check is a clinical or visual review.

## Environments and resources

The historical full environment is preserved in `requirements/rebuild-linux.lock.txt`.
The public package pins the numerical libraries and Torch/XRV versions used by
the model. `requirements/public-linux.lock.txt` records the independently
installed and verified release environment. For a strict Linux/CUDA replay,
install Torch from the CUDA 12.4 index first, then the public lock file, then
install this project with `--no-deps -e .`.

Image training used RTX 5880 48 GB; the original batch-256 profile reserved
approximately 24 GiB. Stage-two training uses 12 CPU threads and keeps the
fit/dev feature matrices in RAM; allow at least 16 GB RAM, with 32 GB preferred.
Retain at least 60–100 GB free disk for caches, exported features, intermediate
checkpoints and source archives. Actual runtime depends on storage and hardware.

## Package an already completed training run

If training has completed and you need another deployment copy, packaging can
be repeated into a fresh directory without repeating optimization:

```bash
python scripts/public/package_run.py --run runs/reproduction --output weights/retrained
```

This verifies completed image and readout outputs, their SHA256 values and their
shared checkpoint identity. It does not resume interrupted training.

## Three different reproduction claims

1. **Use the published model:** download the fixed revision, verify SHA values,
   run `predict.py`. The bundle includes its fitted normalizer and requires no
   historical paths or source data.
2. **Recompute evaluation:** acquire the fixed source datasets, rebuild the
   cache and run `evaluate.py` with published weights. Source image, presentation
   and query denominators remain explicit. A small CPU technical subset is not
   a scientific test result.
3. **Retrain:** run the full two-stage pipeline from the official initialization.
   This reproduces the declared recipe and developer-set selection rules.
   A fixed seed does not guarantee bitwise equality across hardware, kernels,
   precision, batching or library builds. Freshly trained weight hashes are
   therefore recorded separately from the published paper weights.

The public evaluation's percentile intervals use the same grouping principle
and 1,000 draws, but do not claim to replay every draw from the historical
multi-method comparison's random stream. Point-estimate reproduction and
historical paired-baseline confidence intervals are separate claims.

## Validation evidence

`reproducibility/release_validation.json` records the checks actually completed
for this release, with scope and remaining limitations. `reference_metrics.json`
contains the frozen paper-model point estimates, with provenance hashes.
Do not interpret a unit-test count as a fresh full-dataset training result.

## Report review and benchmark scope

The learned OA scoring method is fully represented by the released image model,
readout and region query definitions. The historical report study additionally
used externally licensed MAIRA-2, statement eligibility rules and fixed review
budgets. The image-model release does not mirror restricted generator weights
or automatically reproduce all historical baseline/report/Robust/LOSO tables.
Original statement extraction and geometric metric source code are included
for inspection. Full baseline comparisons must also obtain and run their own
independently licensed assets.

## Limits that remain after publishing

All original fitting used seed 17; grouped sample bootstrap does not measure
training-seed variability. COVIDQU patient mapping is unknown, Kermany families
are filename-derived, some evaluation sources had historical exposure, and NIH
uses pseudo masks. Scores describe relative anatomy, not clinical correctness.
No clinical deployment or verified cross-source patient independence is claimed.
