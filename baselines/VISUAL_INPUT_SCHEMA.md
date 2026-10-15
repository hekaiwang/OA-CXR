# Current-image input schema

Each JSONL row has exactly seven nonempty string fields:
`input_id`, `source_id`, `dataset`, `split`, `source_image_sha256`,
`presented_image_path`, `presented_image_sha256`.

The presented PNG is the current model input. Original images, masks, target
values, crop boxes and clinical reports are not accepted as extra input columns.
The source hash is an identity field, not a path to the original image.
All expected inputs are declared before scoring, with unique input IDs and
valid split membership. Relative PNG paths resolve against the input manifest.

Generate these files with `scripts/public/baseline_inputs.py`; retain them
locally for a private cohort. They are not anonymization tools or publishable
private-data exports.
