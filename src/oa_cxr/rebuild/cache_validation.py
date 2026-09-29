"""Separate public automated integrity validation from historical visual review."""
from pathlib import Path
from oa_cxr.io import load_json, sha256_file


def validation_receipt(root):
    root = Path(root)
    public = root / 'validation.json'
    if public.exists():
        value = load_json(public)
        if (value.get('schema') != 'oa-cxr-public-cache-validation-v1'
                or value.get('status') != 'passed'
                or value.get('cache_status_sha256') != sha256_file(root / 'status.json')
                or value.get('source_and_annotation_sha256_checked') is not True
                or value.get('mode') != 'automated_integrity'):
            raise ValueError('Invalid public cache integrity receipt')
        return public
    historical = root / 'visual_review.json'
    value = load_json(historical)
    if (value.get('status') != 'passed' or value.get('inspected') is not True
            or value.get('cache_status_sha256') != sha256_file(root / 'status.json')):
        raise ValueError('actual cache technical visual review must pass and bind the completed status SHA')
    return historical
