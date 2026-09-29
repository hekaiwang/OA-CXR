# OA-CXR

Trainable implementation and public weights for **relative anatomical-retention
estimation in chest radiograph report review**. Given a presented image, OA-CXR
returns basal, peripheral and whole-lung retention scores associated with
pleural effusion, pneumothorax and consolidation queries.

The default released model is the development-selected paper model (seed 17,
image epoch 12, direct linear L1 readout epoch 27). It is not the optional Robust
candidate, and MAIRA-2 is not part of these weights.

- [Public model and reproducibility artifacts](https://huggingface.co/wanghekai/OA-CXR)
- [Data acquisition and exact frozen splits](docs/DATA.md)
- [Training, evaluation and reproduction boundaries](docs/REPRODUCIBILITY.md)
- [Licenses and third-party attribution](THIRD_PARTY_NOTICES.md)
- [中文说明](README.zh-CN.md)

## Install

Run commands from a clone of this repository. The reference environment is Linux,
Python 3.10 and CUDA 12.4. Install into a **new** environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[training,download,dev]'
python -m pip check
python -m pytest -q
```

CPU inference is supported: use the PyTorch `/whl/cpu` index in the Torch command.
The paper image-training stage requires an idle CUDA GPU with BF16 support.
The original batch 256 used approximately 24 GiB reserved GPU memory on an RTX
5880; leave headroom. CPU and CUDA BF16 outputs need not be bitwise identical.

## Download and predict

```bash
python scripts/public/download.py --output weights
python scripts/public/predict.py \
  --weights weights/main --image /path/to/image.png --output runs/example
```

The download script pins an immutable Hugging Face revision and verifies every
file SHA256. Inference requires only the model bundle and the current PNG/JPEG;
no training data, original uncropped image or ground-truth mask is read.
For GPU inference, add `--gpu-uuid GPU-...` using an idle device from `nvidia-smi -L`.

## Train from the official initialization

First obtain the two source archives described in [DATA.md](docs/DATA.md).
The V7-derived annotation masks are separately attributed CC BY 4.0 artifacts.

```bash
python scripts/public/download.py --output weights_training --with-training-masks
python scripts/public/download_initial.py --output weights_initial
python scripts/public/prepare_data.py \
  --covidqu-archive /path/to/covidqu-v7.zip \
  --kermany-archive /path/to/chest-xray-pneumonia-v2.zip \
  --kermany-masks weights_training/kermany_masks \
  --output data/main --workers 8
python scripts/public/train_pipeline.py \
  --cache data/main/cache --initial weights_initial/initial_state.pt \
  --gpu-uuid GPU-... --output runs/reproduction
```

This executes the image smoke check, **13 full image epochs**, fit/dev feature
export, label attachment, **three readout candidates × 30 epochs**, development
selection and deployment packaging. It does not use calibration/test for fitting
or checkpoint selection. `runs/reproduction/deploy` works with the prediction
and evaluation commands. Failed/partial outputs are retained; use a fresh output
directory for a new attempt. Exact interrupted optimizer continuation is not
implemented.

## Evaluate

```bash
python scripts/public/evaluate.py \
  --weights weights/main --cache data/main/cache --split test \
  --gpu-uuid GPU-... --batch-size 64 --bootstrap 1000 --output runs/test
```

The main test split contains 2,792 source images, 36,296 presentations and
108,888 queries. Evaluation saves every prediction and reports per-source
MAE, RMSE, Spearman, AUROC, AP, AURC and grouped MAE/RMSE intervals. See the
reproduction guide for external datasets and interpretation of numerical drift.

## Scope and limitations

- Labels measure retention **relative to the source image's visible annotated
  anatomy**. Retention 1 does not certify that an image was clinically complete.
- This is not a disease classifier or a clinically validated report-correctness
  detector. The 0.9 review rule is a geometric research rule.
- All reported training used seed 17. Bootstrap intervals describe fixed-model
  sample uncertainty, not repeated-training variability.
- Kermany results are exploratory following historical exposure; filename
  families are not verified cross-dataset patient identities. NIH labels are
  model-generated masks. No claim of zero foundation-pretraining overlap is made.
- This release focuses on the OA method. Historical benchmark orchestration,
  all baseline weights, report-generation outputs and the entire local research
  workspace are not prerequisites for training this implementation.

Original project code is MIT licensed. Model initialization, data and annotation
licenses remain distinct; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
