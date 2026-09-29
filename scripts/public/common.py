"""Shared paths and strict file handling for the portable release commands."""
from pathlib import Path
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts/rebuild')]
from oa_cxr.io import sha256_file, write_json, read_jsonl, load_json


def frozen_rows(name='main'):
    path = ROOT / 'reproducibility' / (name + '_sources.jsonl.gz')
    inventory = load_json(ROOT / 'reproducibility' / 'inventory.json')
    if sha256_file(path) != inventory['files'][path.name]['sha256']:
        raise ValueError('Frozen split manifest SHA differs')
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def checked(path, digest):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError('Missing file or SHA256 mismatch: ' + str(path))
    return path
