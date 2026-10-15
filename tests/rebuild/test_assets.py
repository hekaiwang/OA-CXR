import base64
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rebuild_assets_fixture", Path(__file__).resolve().parents[2] / "scripts/rebuild/assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def fixture_manifest(payload=b"fixed weight bytes"):
    return {"method": "fixture", "asset_root": "model/fixture", "files": {
        "weights.bin": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "url": "https://example.test/weights.bin"}},
        "license_provenance": {"notice": "synthetic fixture"}}


@pytest.mark.parametrize("path", ["../weights", "/absolute", "a/../b", "a\\b", "C:/weights", "a//b", "./weights"])
def test_asset_paths_reject_traversal_and_noncanonical_forms(path):
    with pytest.raises(ValueError):
        assets.relative(path)


def test_download_plan_does_not_guess_unmapped_release_archives():
    manifest = fixture_manifest()
    del manifest["files"]["weights.bin"]["url"]
    plan = assets.download_plan(manifest)
    assert len(plan["blocked"]) == 1 and plan["files"] == []
    manifest.update(upstream="https://huggingface.co/owner/explicit-model", revision="a" * 40)
    item = assets.download_plan(manifest)["files"][0]
    assert item["repository"] == "owner/explicit-model" and item["revision"] == "a" * 40
    assert item["filename"] == "weights.bin" and "url" not in item


def test_staging_cannot_overwrite_runtime_asset_or_existing_attempt(tmp_path):
    with pytest.raises(ValueError, match="asset_staging child"):
        assets.staging(tmp_path / "baselines/chexpert/checkpoints/pretrained", tmp_path)
    target = tmp_path / "asset_staging/attempt"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        assets.staging(target, tmp_path)
    with pytest.raises(FileExistsError):
        assets.staging(tmp_path / "asset_staging/attempt/../other", tmp_path)


def test_verify_checks_declared_hash_and_length(tmp_path):
    payload = b"fixed weight bytes"
    path = tmp_path / "model/fixture/weights.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = fixture_manifest(payload)
    assert assets.verify(manifest, tmp_path)["files"] == 1
    path.write_bytes(payload[:-1] + b"!")
    with pytest.raises(ValueError, match="Asset bytes differ"):
        assets.verify(manifest, tmp_path)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.test/final"


def test_download_uses_exact_roster_and_only_emits_unbound_candidate(tmp_path, monkeypatch):
    payload = b"fixed weight bytes"
    monkeypatch.setattr(assets, "build_opener", lambda *args: type("Opener", (), {"open": lambda self, *a, **kw: Response(payload)})())
    manifest = fixture_manifest(payload)
    out = tmp_path / "asset_staging/fixture"
    receipt = assets.download(manifest, {"manifest_sha256": "b" * 64}, out, tmp_path)
    result = json.loads((out / "candidate_manifest.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["runtime_configs_modified"] is False and receipt["original_assets_modified"] is False
    assert result["release_status"] == "candidate_only_not_bound_to_any_run"
    assert result["license_provenance"] == manifest["license_provenance"]
    assert result["files"]["weights.bin"]["sha256"] == manifest["files"]["weights.bin"]["sha256"]
    assert not (tmp_path / "model").exists()
    with pytest.raises(FileExistsError):
        assets.download(manifest, {"manifest_sha256": "b" * 64}, out, tmp_path)


def test_changed_download_or_missing_provenance_cannot_produce_completion(tmp_path, monkeypatch):
    manifest = fixture_manifest()
    monkeypatch.setattr(assets, "build_opener", lambda *args: type("Opener", (), {"open": lambda self, *a, **kw: Response(b"wrong")})())
    out = tmp_path / "asset_staging/corrupt"
    with pytest.raises(ValueError, match="length/SHA mismatch"):
        assets.download(manifest, {"manifest_sha256": "b" * 64}, out, tmp_path)
    assert not (out / "receipt.json").exists() and not (out / "candidate_manifest.json").exists()
    del manifest["files"]["weights.bin"]["url"]
    other = tmp_path / "asset_staging/missing"
    with pytest.raises(ValueError, match="incomplete download provenance"):
        assets.download(manifest, {"manifest_sha256": "b" * 64}, other, tmp_path)
    assert not other.exists()


@pytest.mark.parametrize("url", ["http://example.test/file", "file:///secret", "https://user:password@example.test/file"])
def test_unencrypted_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        assets.https(url)


def test_converter_refuses_untrusted_pickle_before_torch_import(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"untrusted pickle")
    manifest = {"method": "chexpert", "upstream": assets.LEGACY_URL, "files": {
        assets.LEGACY_FILE: {"bytes": assets.LEGACY_BYTES, "sha256": assets.LEGACY_SHA}}}
    monkeypatch.setattr(assets.importlib.metadata, "version", lambda name: pytest.fail("Untrusted bytes must fail before importing model runtime"))
    with pytest.raises(ValueError, match="Asset bytes differ"):
        assets.convert_chexpert(manifest, {"manifest_sha256": "b" * 64}, source, tmp_path / "asset_staging/conversion", tmp_path)
    assert not (tmp_path / "asset_staging").exists()


def test_chexpert_plan_downloads_original_and_declares_conversion():
    manifest = {"method": "chexpert", "upstream": assets.LEGACY_URL, "files": {
        assets.LEGACY_FILE: {"bytes": assets.LEGACY_BYTES, "sha256": assets.LEGACY_SHA},
        assets.TENSOR_FILE: {"bytes": 1, "sha256": "c" * 64},
        assets.TENSOR_FILE + ".json": {"bytes": 1, "sha256": "d" * 64}}}
    plan = assets.download_plan(manifest)
    assert not plan["blocked"] and len(plan["files"]) == 1
    assert plan["files"][0]["url"] == assets.LEGACY_URL
    assert len(plan["generated_by_convert_chexpert"]) == 2


def test_recipe_cannot_change_bound_weight_identity():
    manifest = fixture_manifest()
    manifest["download_recipes"] = {"weights.bin": {"provider": "literal_utf8", "text": "wrong",
        "bytes": 5, "sha256": hashlib.sha256(b"wrong").hexdigest()}}
    with pytest.raises(ValueError, match="changes the bound asset identity"):
        assets.download_plan(manifest)


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in entries:
            member = tarfile.TarInfo(name)
            if isinstance(value, bytes):
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = value
                archive.addfile(member)


def selected_record(payload=b"official"):
    return {"path": "radgraph/weights.th", "member": "weights.th", "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def test_archive_extracts_only_declared_exact_bytes(tmp_path):
    archive = tmp_path / "model.tar.gz"
    make_archive(archive, [("unused.txt", b"unused"), ("weights.th", b"official")])
    assets._extract_declared(archive, [selected_record()], tmp_path / "files")
    assert (tmp_path / "files/radgraph/weights.th").read_bytes() == b"official"
    assert not (tmp_path / "files/unused.txt").exists()


@pytest.mark.parametrize("entries,match", [
    ([("../outside", b"bad")], "escapes"),
    ([("weights.th", "../outside")], "Unsupported"),
    ([("weights.th", b"official"), ("weights.th", b"official")], "duplicate"),
    ([("other", b"official")], "missing"),
    ([("weights.th", b"tampered")], "SHA mismatch"),
])
def test_archive_rejects_bad_members_and_identity(tmp_path, entries, match):
    archive = tmp_path / "model.tar.gz"
    make_archive(archive, entries)
    with pytest.raises(ValueError, match=match):
        assets._extract_declared(archive, [selected_record()], tmp_path / "files")
    assert not (tmp_path / "outside").exists()


def test_github_blob_requires_both_git_and_bound_sha(tmp_path, monkeypatch):
    raw = b"fixed public source"
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    blob = {"encoding": "base64", "sha": git_sha, "content": base64.b64encode(raw).decode()}
    item = {"url": "https://api.github.com/repos/official/project/git/blobs/" + git_sha,
            "git_blob_sha1": git_sha, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(assets, "build_opener", lambda *args: type("Opener", (), {"open": lambda self, *a, **kw: Response(json.dumps(blob).encode())})())
    assets._download_github_blob(item, tmp_path / "source.py")
    assert (tmp_path / "source.py").read_bytes() == raw
    blob["content"] = base64.b64encode(b"changed source").decode()
    with pytest.raises(ValueError, match="Git object SHA"):
        assets._download_github_blob(item, tmp_path / "changed.py")
    assert not (tmp_path / "changed.py").exists()


def test_cache_symlink_cannot_escape_staging(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "snapshot").mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"official")
    item = {**selected_record(), "path": "snapshot/weights.th", "target": "../../outside",
            "resolved_relative": "../outside"}
    with pytest.raises(ValueError, match="escapes"):
        assets._link_declared(item, folder)


def test_verified_source_recipes_cover_all_bound_files():
    for method in ("scuq_source", "scuq_radgraph"):
        manifest, binding = assets.bound_manifest(method)
        plan = assets.download_plan(manifest)
        assert not plan["blocked"]
        assert {x["path"] for x in plan["files"]} == set(manifest["files"])
        assert len(binding["download_sources_sha256"]) == 64


def test_hf_download_pins_revision_and_keeps_files_inside_candidate(tmp_path, monkeypatch):
    output = tmp_path / "asset_staging/attempt"
    output.mkdir(parents=True)
    item = {**selected_record(), "repository": "official/encoder", "revision": "a" * 40, "filename": "weights.bin"}
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"official")
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=provider))
    target = output / "files/blob"
    assets._download_hf(item, target, output)
    assert target.read_bytes() == b"official"
    assert calls[0]["revision"] == "a" * 40 and calls[0]["repo_id"] == "official/encoder"
    assert calls[0]["local_dir"].is_relative_to(output)
    item["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        assets._download_hf(item, target, output)


@pytest.mark.skipif(os.name == "nt", reason="HF relative symlink reconstruction targets the Linux server")
def test_hf_snapshot_link_preserves_text_and_resolves_to_verified_blob(tmp_path):
    folder = tmp_path / "files"
    path = folder / "encoder/blobs/object"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"official")
    (folder / "encoder/snapshots/revision").mkdir(parents=True)
    item = {**selected_record(), "path": "encoder/snapshots/revision/weights.bin",
            "target": "../../blobs/object", "resolved_relative": "encoder/blobs/object"}
    assets._link_declared(item, folder)
    link = folder / item["path"]
    assert link.is_symlink() and os.readlink(link) == item["target"]
    assert link.resolve() == path and link.read_bytes() == b"official"
