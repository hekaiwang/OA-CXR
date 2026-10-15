"""Verify pinned assets, or build a fresh staging candidate without rebinding runs.

Download URLs/repositories come from bound asset metadata. The only unrestricted
Torch load is the byte-exact, fixed official legacy CheXpert release below.
"""
from __future__ import annotations
import argparse
import base64
from collections import OrderedDict
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oa_cxr.io import exclusive_writer, load_json, sha256_file, write_json

LEGACY_FILE = "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
LEGACY_URL = "https://github.com/mlmed/torchxrayvision/releases/download/v1/" + LEGACY_FILE
LEGACY_SHA = "1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4"
LEGACY_BYTES = 28381998
TENSOR_FILE = "chexpert_tensor_state.pt"
METHODS = ("chexpert", "autoqc", "biovil_t", "maira_shared", "scuq_radgraph", "scuq_source")


def relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Invalid asset relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("..", ".") for p in name.split("/")) or "" in name.split("/"):
        raise ValueError("Asset path escapes its declared root")
    return path


def roster(manifest):
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Explicit nonempty asset file roster required")
    for name, record in files.items():
        relative(name)
        if (not isinstance(record, dict) or type(record.get("bytes")) is not int or record["bytes"] < 0 or
                not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise ValueError("Every asset needs exact byte length and SHA256")
    return files


def bound_manifest(method, root=ROOT):
    folder = root / "baselines" / ("maira_logprob" if method == "maira_shared" else "scuq_radgraph" if method == "scuq_source" else method)
    config = folder / ("generation_config.json" if method == "maira_shared" else "config.json")
    setting = load_json(config)["source_manifest" if method == "scuq_source" else "assets_manifest"]
    path = (folder / setting["path"]).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()) or sha256_file(path) != setting["sha256"]:
        raise ValueError("Asset manifest/config SHA binding differs")
    manifest = load_json(path)
    roster(manifest)
    binding = {"config_sha256": sha256_file(config), "manifest_sha256": setting["sha256"], "manifest_path": str(path)}
    source_path = root / "scripts/rebuild/asset_sources.json"
    if method in ("scuq_source", "scuq_radgraph") and source_path.is_file():
        sources = load_json(source_path)
        source = sources["bindings"][method]
        if sources.get("schema") != "oa-cxr-verified-download-sources-v1" or source["manifest_sha256"] != binding["manifest_sha256"]:
            raise ValueError("Download provenance is not bound to the current runtime manifest")
        manifest = {**manifest, "download_recipes": source["recipes"], "download_archives": source.get("archives", {})}
        binding["download_sources_sha256"] = sha256_file(source_path)
    return manifest, binding


def verify_file(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("Asset bytes differ: " + path.name)


def verify(manifest, root=ROOT):
    folder = (root / str(relative(manifest["asset_root"]))).resolve(strict=True)
    if not folder.is_relative_to(root.resolve()):
        raise ValueError("Asset root leaves project")
    for name, record in roster(manifest).items():
        path = (folder / name).resolve(strict=True)
        if not path.is_relative_to(folder):
            raise ValueError("Asset symlink leaves its bound root")
        verify_file(path, record)
    return {"status": "verified", "files": len(manifest["files"]), "asset_root": str(folder)}


def https(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only HTTPS asset URLs without embedded credentials are accepted")
    return url


def download_plan(manifest):
    planned, blocked, generated = [], [], []
    upstream = manifest.get("upstream", "")
    hf = re.fullmatch(r"https://huggingface\.co/([^/]+/[^/]+)/?", upstream)
    revision = manifest.get("revision", "")
    for name, record in roster(manifest).items():
        recipe = manifest.get("download_recipes", {}).get(name)
        if recipe is not None:
            if recipe.get("bytes") != record["bytes"] or recipe.get("sha256") != record["sha256"]:
                raise ValueError("Download recipe changes the bound asset identity")
            source = {k: v for k, v in recipe.items() if k not in ("bytes", "sha256")}
            if source.get("provider") not in ("github_blob", "huggingface", "archive_member", "literal_utf8", "symlink"):
                raise ValueError("Unsupported explicit download recipe")
        elif record.get("url"):
            source = {"provider": "https", "url": https(record["url"])}
        elif (manifest.get("method") == "chexpert" and name == LEGACY_FILE and
              upstream == LEGACY_URL and record == {"bytes": LEGACY_BYTES, "sha256": LEGACY_SHA}):
            source = {"provider": "https", "url": LEGACY_URL}
        elif manifest.get("method") == "chexpert" and name in (TENSOR_FILE, TENSOR_FILE + ".json"):
            generated.append(name)
            continue
        elif hf and re.fullmatch(r"[0-9a-f]{40}", revision):
            # Repository, revision and filename are declared metadata, not inferred URLs.
            source = {"provider": "huggingface", "repository": hf.group(1), "revision": revision, "filename": name}
        else:
            blocked.append({"path": name, "reason": "No declared per-file URL or pinned Hugging Face repository mapping; use preserved verified assets until download provenance is supplied"})
            continue
        planned.append({"path": name, "bytes": record["bytes"], "sha256": record["sha256"], **source})
    return {"files": planned, "archives": manifest.get("download_archives", {}),
            "generated_by_convert_chexpert": generated, "blocked": blocked}


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_https(item, destination):
    url = https(item["url"])
    temporary = destination.with_name(destination.name + ".part")
    received = 0
    digest = hashlib.sha256()
    with build_opener(HTTPSRedirects()).open(Request(url, headers={"User-Agent": "oa-cxr-pinned-assets"}), timeout=90) as response, temporary.open("xb") as output:
        https(response.geturl())
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            received += len(block)
            if received > item["bytes"]:
                raise ValueError("Download exceeds declared asset size")
            digest.update(block)
            output.write(block)
    if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise ValueError("Downloaded asset length/SHA mismatch")
    temporary.replace(destination)


def _write_verified(raw, destination, item):
    if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Generated/downloaded asset bytes differ from declared identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)


def _download_hf(item, destination, output):
    from huggingface_hub import hf_hub_download
    filename = str(relative(item["filename"]))
    if not re.fullmatch(r"[0-9a-f]{40}", item["revision"]):
        raise ValueError("Hugging Face revision must be immutable")
    # Use only this staging directory, including provider bookkeeping/cache files.
    namespace = hashlib.sha256((item["repository"] + item["revision"]).encode()).hexdigest()
    folder = output / ".downloads" / namespace
    downloaded = Path(hf_hub_download(repo_id=item["repository"], revision=item["revision"],
                                     filename=filename, local_dir=folder))
    if downloaded.is_symlink() or not downloaded.resolve(strict=True).is_relative_to(output):
        raise ValueError("Hugging Face download left fresh staging directory")
    verify_file(downloaded, item)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded.rename(destination)


def _download_github_blob(item, destination):
    url = https(item["url"])
    if not url.startswith("https://api.github.com/repos/") or not re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha1"]):
        raise ValueError("Expected the explicit official GitHub immutable blob API URL")
    request = Request(url, headers={"User-Agent": "oa-cxr-pinned-assets", "Accept": "application/json"})
    with build_opener(HTTPSRedirects()).open(request, timeout=90) as response:
        payload = response.read(item["bytes"] * 2 + 8193)
    if len(payload) > item["bytes"] * 2 + 8192:
        raise ValueError("GitHub blob response exceeds declared content size")
    blob = json.loads(payload)
    if blob.get("encoding") != "base64" or blob.get("sha") != item["git_blob_sha1"]:
        raise ValueError("GitHub blob identity/encoding mismatch")
    raw = base64.b64decode(blob["content"])
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != item["git_blob_sha1"]:
        raise ValueError("Git object SHA mismatch")
    _write_verified(raw, destination, item)


def _extract_declared(archive, selected, folder):
    required = {str(relative(item["member"])): item for item in selected}
    if len(required) != len(selected):
        raise ValueError("Multiple outputs cannot ambiguously share an archive member")
    found, encountered = set(), set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            relative(member.name)
            if member.name in encountered or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported or duplicate archive member")
            encountered.add(member.name)
            if member.name not in required:
                continue
            item = required[member.name]
            if not member.isfile() or member.size != item["bytes"]:
                raise ValueError("Archive member size/type differs")
            path = folder / str(relative(item["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with stream.extractfile(member) as source, path.open("xb") as destination:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    destination.write(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Extracted asset SHA mismatch")
            found.add(member.name)
    if found != set(required):
        raise ValueError("Declared archive members missing")


def _link_declared(item, folder):
    path = folder / str(relative(item["path"]))
    target = (path.parent / item["target"]).resolve(strict=True)
    expected = (folder / str(relative(item["resolved_relative"]))).resolve(strict=True)
    if target != expected or not target.is_relative_to(folder.resolve()) or path.exists():
        raise ValueError("HF snapshot link does not resolve to its declared internal blob")
    verify_file(target, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(item["target"], path)


def staging(output, root=ROOT):
    root = root.resolve()
    allowed = root / "asset_staging"
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    # Resolve only after rejecting existing symlinks, including the staging root.
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Use a fresh project asset_staging child")
    current = root
    for part in output.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Staging path contains a symlink")
    if output.resolve() != output or output.exists():
        raise FileExistsError("Staging output must be fresh and canonical")
    return output


def candidate(manifest, output, files, binding, action, root=ROOT, **extra):
    result = copy.deepcopy(manifest)
    result.update(asset_root=str((output / "files").relative_to(root.resolve())).replace("\\", "/"), files=files,
                  release_status="candidate_only_not_bound_to_any_run", parent_manifest_sha256=binding["manifest_sha256"], **extra)
    write_json(output / "candidate_manifest.json", result)
    verify(result, root)
    receipt = {"status": "completed", "action": action, "created_at": time.time(), "source_binding": binding,
               "candidate_manifest_sha256": sha256_file(output / "candidate_manifest.json"),
               "original_assets_modified": False, "runtime_configs_modified": False, "files": files}
    write_json(output / "receipt.json", receipt)
    return receipt


def download(manifest, binding, output, root=ROOT):
    plan = download_plan(manifest)
    if plan["blocked"]:
        raise ValueError("Manifest has incomplete download provenance; inspect the plan before staging")
    output = staging(output, root)
    with exclusive_writer(output):
        output.mkdir()
        write_json(output / "download_plan.json", plan)
        files = {}
        for archive_id, archive_item in plan["archives"].items():
            if not re.fullmatch(r"[a-z0-9_]+", archive_id):
                raise ValueError("Invalid archive identifier")
            selected = [item for item in plan["files"] if item["provider"] == "archive_member" and item["archive"] == archive_id]
            path = output / "archives" / (archive_id + ".tar.gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            if archive_item["provider"] == "huggingface":
                _download_hf(archive_item, path, output)
            elif archive_item["provider"] == "https":
                _download_https(archive_item, path)
            else:
                raise ValueError("Unsupported pinned archive source")
            verify_file(path, archive_item)
            _extract_declared(path, selected, output / "files")
        for item in plan["files"]:
            path = output / "files" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if item["provider"] == "https":
                _download_https(item, path)
            elif item["provider"] == "huggingface":
                _download_hf(item, path, output)
            elif item["provider"] == "github_blob":
                _download_github_blob(item, path)
            elif item["provider"] == "literal_utf8":
                raw = item["text"].encode("utf-8")
                if len(raw) > 4096:
                    raise ValueError("Only small pinned cache references may be generated")
                _write_verified(raw, path, item)
            elif item["provider"] == "symlink":
                continue
            elif item["provider"] != "archive_member":
                raise ValueError("Unknown asset provider")
            verify_file(path, item)
            files[item["path"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in (x for x in plan["files"] if x["provider"] == "symlink"):
            _link_declared(item, output / "files")
            files[item["path"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
        result = candidate(manifest, output, files, binding, "download", root,
                           conversion_still_required=bool(plan["generated_by_convert_chexpert"]))
        verify(load_json(output / "candidate_manifest.json"), root)
        return result


def tensor_state(payload, torch):
    if isinstance(payload, torch.nn.Module):
        for module in payload.modules():
            if not hasattr(module, "_non_persistent_buffers_set"):
                module._non_persistent_buffers_set = set()
        payload = payload.state_dict()
    if not isinstance(payload, dict) or not payload or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in payload.items()):
        raise ValueError("Expected a nonempty tensor-only state dictionary")
    prefixed = [k.startswith("module.") for k in payload]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed DataParallel prefixes")
    return OrderedDict((k[7:] if all(prefixed) else k, v.detach().cpu()) for k, v in payload.items())


def convert_chexpert(manifest, binding, source, output, root=ROOT):
    record = roster(manifest).get(LEGACY_FILE)
    if manifest.get("method") != "chexpert" or record != {"bytes": LEGACY_BYTES, "sha256": LEGACY_SHA} or manifest.get("upstream") != LEGACY_URL:
        raise ValueError("Conversion requires the fixed official CheXpert source identity")
    source = Path(source).resolve(strict=True)
    if not source.is_relative_to(root.resolve()):
        raise ValueError("Conversion source must be inside the project")
    verify_file(source, record)
    output = staging(output, root)
    if importlib.metadata.version("torchxrayvision") != "1.5.4":
        raise ValueError("Conversion requires torchxrayvision 1.5.4")
    import torch
    import torchxrayvision as xrv
    if torch.__version__.split("+")[0] != "2.6.0":
        raise ValueError("Conversion requires the recorded Torch 2.6.0 serializer")
    with exclusive_writer(output):
        output.mkdir()
        folder = output / "files"
        folder.mkdir()
        copied = folder / LEGACY_FILE
        shutil.copyfile(source, copied)
        verify_file(copied, record)
        # No configurable hash or unsafe fallback: this one exact upstream byte stream only.
        state = tensor_state(torch.load(copied, map_location="cpu", weights_only=False), torch)
        reference = xrv.models.DenseNet(weights=None, op_threshs=None, apply_sigmoid=False)
        reference.load_state_dict(state, strict=True)
        target = folder / TENSOR_FILE
        torch.save(state, target)
        reloaded = torch.load(target, map_location="cpu", weights_only=True)
        reference.load_state_dict(reloaded, strict=True)
        if list(state) != list(reloaded) or any(not torch.equal(state[k], reloaded[k]) for k in state):
            raise ValueError("Tensor values changed during conversion")
        conversion = {"source_url": LEGACY_URL, "source_sha256": LEGACY_SHA, "sha256": sha256_file(target),
                      "bytes": target.stat().st_size, "torch": str(torch.__version__), "torchxrayvision": "1.5.4",
                      "converted_at_unix": time.time(), "upstream_signed_hash": False,
                      "policy": "fixed official SHA-only legacy conversion; strict architecture and tensor-only reload checked",
                      "tensor_count": len(state), "all_tensor_values_verified": True}
        write_json(folder / (TENSOR_FILE + ".json"), conversion)
        files = {name: {"bytes": (folder / name).stat().st_size, "sha256": sha256_file(folder / name)}
                 for name in (LEGACY_FILE, TENSOR_FILE, TENSOR_FILE + ".json")}
        verify_file(source, record)
        result = candidate(manifest, output, files, binding, "convert-chexpert", root,
                           conversion=conversion, conversion_still_required=False)
        verify(load_json(output / "candidate_manifest.json"), root)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "plan", "download", "convert-chexpert"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    manifest, binding = bound_manifest(args.method)
    if args.action == "verify":
        result = {**verify(manifest), "source_binding": binding}
    elif args.action == "plan":
        result = {**download_plan(manifest), "source_binding": binding}
    else:
        if args.output is None:
            parser.error("Staging actions require --output asset_staging/NEW_NAME")
        if args.action == "download":
            result = download(manifest, binding, args.output)
        else:
            if args.method != "chexpert":
                parser.error("convert-chexpert requires --method chexpert")
            source = args.source or ROOT / manifest["asset_root"] / LEGACY_FILE
            result = convert_chexpert(manifest, binding, source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
