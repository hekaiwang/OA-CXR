# Private clinical cohort and data availability

The study also uses a **private clinical cohort that is not publicly available**.
Reference lung regions were **manually delineated by senior physicians**.
This annotation provenance is supplied by the study authors. The public release
does not independently audit the private annotations.

The repository provides code, input-format specifications and evaluation
protocols. It does not distribute private radiographs, lung masks, clinical
reports, patient identifiers, source-file manifests, linkage tables or
record-level predictions. Only aggregate results may be reported. Private data
are not available through this repository, and no access-on-request promise is
made. The four downloadable public datasets are separate from this cohort.

## Suggested data availability statement

> Code and OA-CXR model weights are publicly available. The public benchmark
> datasets can be obtained from their respective providers. The private clinical
> dataset cannot be publicly released because of patient privacy and institutional
> confidentiality restrictions. Its reference lung regions were manually
> delineated by senior physicians. We provide the data interface, preprocessing
> and evaluation code, and report aggregate study results. Reproduction of the
> private-cohort numerical results requires authorized access to the original
> data and annotations.

The manuscript provides the study-specific ethics and consent statement.

## Local interface for authorized users

`scripts/public/prepare_private.py` accepts de-identified PNG/JPEG source images
and two native-resolution binary masks per source, supplied locally by an
authorized user. Masks must be nonempty, disjoint and ordered by **image-coordinate
left/right**, not clinical laterality. The command does not anonymize images or
remove burned-in text. Use already de-identified exports; keep the identity
mapping outside the repository.

The manifest has exactly these top-level fields:

```json
{
  "schema": "oa-cxr-restricted-cohort-v1",
  "dataset": "private_clinical",
  "annotation_type": "senior_physician_manual_lung_masks",
  "samples": [
    {
      "source_id": "case_0123456789abcdef",
      "group_id": "group_fedcba9876543210",
      "image": "images/example.png",
      "image_sha256": "REPLACE_WITH_SHA256",
      "image_left_mask": "masks/example_left.png",
      "image_left_mask_sha256": "REPLACE_WITH_SHA256",
      "image_right_mask": "masks/example_right.png",
      "image_right_mask_sha256": "REPLACE_WITH_SHA256"
    }
  ]
}
```

The example contains no real patient or study record. Generate random case/group
aliases, use the same group alias for related images, and compute the actual
file checksums locally. File paths are relative to the local manifest directory.
No names, dates, clinical notes or extra metadata fields are accepted.

```bash
python scripts/public/prepare_private.py \
  --manifest private_data/manifest.json --output private_data/cache
python scripts/public/evaluate.py --weights weights/main \
  --cache private_data/cache --split external_private \
  --gpu-uuid GPU-... --output local_results/private_oa
```

This builds the shared 13-presentation/three-query geometric protocol using
native-resolution manual masks before resizing. An invalid annotation stops
preparation; no model-derived mask is substituted. Image inference reads only
the presented image. Source masks are used to compute independent targets.

The same cache can be passed to the baseline feature export, image materialization
and comparison commands with `--split external_private`. Use frozen models fitted
on the public fit/dev data; private evaluation must not select checkpoints or
thresholds. Scores still measure anatomical retention, not disease correctness.

## Reporting and verification boundaries

Keep individual images, masks, source IDs/hashes, predictions and bootstrap
membership local. `private_data/`, `restricted_data/` and `local_results/` are
ignored by Git. Review any proposed aggregate table separately before publishing
it; the software does not upload local evaluation outputs.

Report source-image and query denominators, grouping rules, missing predictions,
and uncertainty together with aggregate metrics. Do not call a grouping count a
verified independent-patient count unless that has been established. The public
machine-readable reference table covers the public benchmarks only. Private-cohort
point estimates must correspond to the manual-mask protocol and the final results
reported in the manuscript.

Readers without the restricted data can run all released algorithms on the
public benchmarks or their own authorized data. That verifies the method and
implementation, but does not independently reproduce the private numerical
results. Synthetic interface tests are explicitly not clinical validation.
