# Code, weights and data have separate provenance

## Project code

Original OA-CXR code is distributed under the repository's existing MIT license,
copyright 2026 hekaiwang. Dependencies retain their own licenses; installing them
does not relicense them as MIT. No MAIRA-2 implementation or checkpoint is copied
into this release.

## OA-CXR weights and TorchXRayVision initialization

The public model consists of the author's fine-tuned image model and trained
readout. It starts from TorchXRayVision's official CheXpert-only DenseNet121
checkpoint, then trains on COVID-QU-Ex and Kermany/V7. The released tensor files
are the exact development-selected OA artifacts, not a renamed baseline.

Upstream: [TorchXRayVision](https://github.com/mlmed/torchxrayvision).
The upstream [license declaration](https://github.com/mlmed/torchxrayvision/blob/main/LICENSE)
identifies its main `xrv.models` library as Apache-2.0 and distinguishes separately
licensed `baseline_models`. OA uses the main `xrv.models.DenseNet`.
The official release checkpoint did not include a separate weight-specific
license document; we preserve that provenance limitation and do not represent
the entire pretrained model or training data as independently authored MIT work.

The Hugging Face model card and weight-license notice distinguish author-owned
contributions, upstream rights and intended research use. The original XRV
checkpoint is fetched directly from upstream by `download_initial.py`, after
verification against a fixed SHA, rather than mirrored here.

Reference: Cohen et al., *TorchXRayVision: A library of chest X-ray datasets and
models*, [arXiv:2111.00595](https://arxiv.org/abs/2111.00595).

## V7-derived annotation masks

The optional `reproduction/kermany-v7-masks.tar.gz` contains rasterized masks
derived from V7/CloudFactory human-assisted and human-reviewed lung polygons.
The [upstream attribution/license section](https://github.com/v7labs/covid-19-xray-dataset#data-sources-and-licenses)
links [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
We retain that license, credit V7 and CloudFactory, and identify the adaptation:
polygons rendered as two image-coordinate lung PNG masks and associated with
the frozen Kermany cohort. The mask archive does not contain radiographs or
the original metadata's image-access URLs.

Source radiographs: Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018),
*Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for
Classification*, Mendeley Data v2, [doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2).

## Other datasets

COVID-QU-Ex images and annotations are obtained directly from the version-7
provider. NIH images, CheXmask annotations and Shenzhen images/masks are also
obtained under their individual provider terms. The repository supplies fixed
membership and reconstruction code, not a blanket redistribution license for
those datasets. See [DATA.md](docs/DATA.md) for exact sources and versions.

## MAIRA-2 and report experiments

The historical report-generation experiments used frozen MAIRA-2 revision
`795a2b1cd4a310624b4e3d14b5a23e41fd273deb`, greedy BF16 generation with a 512-token
limit. MAIRA-2 is governed by the Microsoft Research License Terms (MSRLA),
which prohibit redistribution of the supplied model. Users seeking to repeat
the report-generation stage must obtain it through
[Microsoft's official model page](https://huggingface.co/microsoft/maira-2)
and comply with those terms. The OA image training and inference commands do
not require MAIRA-2.
