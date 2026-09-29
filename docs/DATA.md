# Data and fixed membership

Raw radiographs are not redistributed here. Obtain source datasets under their
provider terms. `reproducibility/main_sources.jsonl.gz` and
`external_sources.jsonl.gz` contain source IDs, public filenames, image/annotation
SHA256 values, source/family groups and fixed split assignments. They contain no
pixels, image thumbnails, private access tokens or server paths.

| Dataset | fit | dev | calibration | test / external |
|---|---:|---:|---:|---:|
| COVID-QU-Ex | 16,192 | 2,255 | 2,289 | 1,855 |
| Kermany/V7 | 2,894 | 354 | 362 | 937 |
| NIH/CheXmask | — | — | — | 5,821 |
| Shenzhen | — | — | — | 192 |

The JSONL manifests are authoritative if comparing counts. Every source has
13 controlled presentations and three regional queries. These are not additional
independent patients.

## Main training sources

1. **COVID-QU-Ex, version 7.** [Official dataset](https://www.kaggle.com/dsv/3122958).
   Download endpoint:
   `https://www.kaggle.com/api/v1/datasets/download/anasmohammedtahir/covidqu?datasetVersionNumber=7`.
   Archive SHA256:
   `2a91b372cdd104d05d472f79dc446dc9bb7e3ccc26f513cf971f64cabc151333`.
   The preparation script reads the lung segmentation images and masks from
   `Lung Segmentation Data/Lung Segmentation Data/` within this ZIP.
2. **Kermany, Zhang and Goldbaum (2018), version 2.**
   [Mendeley source](https://data.mendeley.com/datasets/rscbjbr9sj/2),
   [Kaggle mirror](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
   Exact mirror endpoint:
   `https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia?datasetVersionNumber=2`.
   ZIP SHA256:
   `f569fe885b0f921e836f3d6bcc8d7b3442f5e0ca4db4533d06b8cf25d2114ea1`.
   Original basenames may occur more than once in the ZIP; the loader selects by
   expected image SHA, not first filename match.
3. **V7/CloudFactory lung annotations.**
   [Upstream annotations and attribution](https://github.com/v7labs/covid-19-xray-dataset),
   [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
   `download.py --with-training-masks` supplies the exact derived, rasterized
   two-channel PNG masks used in this experiment. These masks are adaptations
   of V7 human-assisted, human-reviewed polygons, not new manual annotations.
   Original radiographs and V7 metadata download URLs are not included.

The preparation command validates every selected source image and annotation,
recreates aspect-preserving letterboxing and all geometric targets, and writes
memory-mapped caches. Allow about 40 GB for the main cache plus the archives and
extracted images. Start with `--smoke-per-dataset 2 --splits fit dev test` for a
small technical check; this output deliberately cannot pass full training's
source-count requirements.

## What the frozen splits mean

The original audit considered 31,679 candidates and retained 27,138 after 4,541
COVIDQU duplicate exclusions. It checked bytes, decoded RGB, grayscale hashes
and the declared dHash/thumbnail near-duplicate rule. Main partition intersections
are empty for source ID, source-image SHA and conservative family groups.

`scripts/rebuild/prepare_data.py` retains the original audit algorithm for
inspection. The **public** preparation entrypoint replays the published final
membership, so users do not need the author's historical source bank. Re-running
the original historical-exposure audit itself requires the historical inputs;
that is distinct from recreating the paper's fixed train/evaluation data.

The image normalization is `(gray/255*2-1)*1024`; resizing is bilinear for images
and nearest-neighbor for masks. Empty predicted masks never remove a sample.
Targets use native source annotations before resize, as implemented in
`src/oa_cxr/rebuild/data.py` and `anatomy_geometry.py`.

## External evaluation

**NIH:** obtain the images in the fixed external manifest from
[NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC).
The historical public mirror uses
`https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/<filename>`.
Obtain `OriginalResolution/ChestX-Ray8.csv` from
[CheXmask 1.0.0](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/).
The exact CSV used has SHA256
`48766ab0268235d63666bb2bacbd9f642b33fce7c1be40b9e1ecb381605545fa`.
These are HybridGNet pseudo masks, not human ground truth.

```bash
python scripts/public/prepare_external.py --split external_nih \
  --nih-images /path/to/nih/images --chexmask-csv /path/to/ChestX-Ray8.csv \
  --output data/nih
```

**Shenzhen:** obtain PNG images from
[NLM's Shenzhen distribution](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/index.html)
and the masks from [SHCXR lung masks](https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask), version 1.
The mask ZIP SHA256 is
`4180afa70044cca10fa2b2c2e1a0e458d5b1bb5c0a653c2e1344d83138d53f72`.
Extract the archive yourself and supply its mask directory.

```bash
python scripts/public/prepare_external.py --split external_shenzhen \
  --shenzhen-images /path/to/shenzhen/images --shenzhen-masks /path/to/shenzhen/masks \
  --output data/shenzhen
python scripts/public/evaluate.py --weights weights/main \
  --cache data/shenzhen/cache --split external_shenzhen --gpu-uuid GPU-... \
  --output runs/shenzhen
```

Both external preparation paths verify the exact original mask PNG SHA after
decoding/resolution. Different dataset or annotation versions fail explicitly.
