"""Loader for the official Flybody flight policy released by HHMI Janelia/DeepMind."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import numpy as np


POLICY_ARCHIVE_URL = "https://janelia.figshare.com/ndownloader/files/44815195"
POLICY_ARCHIVE_SHA256 = "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582"
POLICY_ARCHIVE_BYTES = 6_537_720


def _download_archive() -> bytes:
    request = urllib.request.Request(
        POLICY_ARCHIVE_URL,
        headers={"User-Agent": "fly-fruit-fly/0.1 official-policy-loader"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != POLICY_ARCHIVE_BYTES or digest != POLICY_ARCHIVE_SHA256:
        raise RuntimeError(
            "Official Flybody policy archive failed integrity verification: "
            f"got {len(payload)} bytes sha256={digest}"
        )
    return payload


def ensure_official_flight_policy(cache_dir: Path) -> Path:
    """Return a verified local copy of the released flight SavedModel.

    Only the `flight/` subtree is extracted. The upstream archive also contains
    walking and vision policies, which this project does not need for flight.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    policy_dir = cache_dir / "official-flybody-policy" / "flight"
    if (policy_dir / "saved_model.pb").is_file() and (
        policy_dir / "variables" / "variables.index"
    ).is_file():
        return policy_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _download_archive()
    target_root = policy_dir.parent
    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as temp_name:
        temp_root = Path(temp_name)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [name for name in archive.namelist() if name.startswith("flight/")]
            if "flight/saved_model.pb" not in members:
                raise RuntimeError("Official archive does not contain flight/saved_model.pb")
            for name in members:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe archive member: {name}")
                destination = temp_root / relative
                if name.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(temp_root / "flight"), str(policy_dir))
    return policy_dir


class _HTTPRangeReader(io.RawIOBase):
    """Seekable read-only ZIP source with bounded, validated HTTP range reads."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, int(size), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid range seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if size == 0:
            return b""
        if size > 1_048_576:
            raise RuntimeError("ZIP range request exceeds the 1 MiB limit")
        end = self.position + size - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}",
                               "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Official asset server did not honor the requested byte range")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Truncated or oversized ZIP range response")
        self.position += size
        return data


WING_PATTERN_METADATA_URL = "https://api.figshare.com/v2/articles/25309105"


def ensure_official_wing_pattern(cache_dir: Path) -> Path:
    """Fetch the exact named wingbeat asset used by the upstream notebook."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "wing_pattern_fmech.npy"
    manifest_path = cache_dir / "wing_pattern_fmech.json"
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]:
            return path
    with urllib.request.urlopen(WING_PATTERN_METADATA_URL, timeout=60) as response:
        metadata = json.load(response)
    matches = [f for f in metadata["files"] if f["name"] == "datasets_flight-imitation.zip"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {path.name}; available: "
                           f"{[f['name'] for f in metadata['files']]}")
    record = matches[0]
    # The wing pattern is a small member of the flight dataset archive.
    # Read ZIP headers and this member over HTTP ranges, not the whole dataset.
    with zipfile.ZipFile(_HTTPRangeReader(record["download_url"], record["size"])) as archive:
        members = [i for i in archive.infolist() if Path(i.filename).name == path.name]
        if len(members) != 1 or members[0].file_size > 1_048_576:
            raise RuntimeError("Expected one small wing-pattern member in the official ZIP")
        payload = archive.read(members[0])  # zipfile verifies the member CRC32.
        member_name = members[0].filename
        member_crc = members[0].CRC
    pattern = np.load(io.BytesIO(payload), allow_pickle=False)
    if pattern.ndim != 2 or pattern.shape[1] != 3 or not np.isfinite(pattern).all():
        raise RuntimeError("Official wing-pattern shape/values are invalid")
    path.write_bytes(payload)
    manifest_path.write_text(json.dumps({
        "article": WING_PATTERN_METADATA_URL, "article_version": metadata.get("version"),
        "file_id": record["id"], "download_url": record["download_url"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "archive_member": member_name, "member_crc32": member_crc,
        "verification": "Member CRC32 and SHA256; full archive MD5 not checked (range retrieval)",
    }, indent=2) + "\n")
    return path


class OfficialFlightPolicy:
    """Deterministic wrapper around the released TensorFlow SavedModel."""

    def __init__(self, policy_dir: Path):
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp  # noqa: F401 - registers distribution types
        except ImportError as exc:
            raise RuntimeError(
                "Official expert evaluation needs the optional TensorFlow runtime. "
                "Install with: pip install -e '.[expert]'"
            ) from exc
        # Released TFP SavedModels used full Python module names. TFP 0.23
        # registers the same compatible TypeSpecs under "tfp.distributions".
        # Add deserialization aliases only; serialization and tensor math retain
        # the current classes. This bridge is tied to the pinned expert runtime.
        from tensorflow.python.framework import type_spec_registry
        for distribution_cls in (
            tfp.distributions.Independent, tfp.distributions.Normal,
            tfp.distributions.MultivariateNormalDiag,
        ):
            modern = f"tfp.distributions.{distribution_cls.__name__}_ACTTypeSpec"
            legacy = f"{distribution_cls.__module__}.{distribution_cls.__name__}_ACTTypeSpec"
            spec = type_spec_registry.lookup(modern)
            existing = type_spec_registry._NAME_TO_TYPE_SPEC.get(legacy)
            if existing is not None and existing is not spec:
                raise RuntimeError(f"Conflicting SavedModel TypeSpec alias: {legacy}")
            type_spec_registry._NAME_TO_TYPE_SPEC[legacy] = spec
        self._tf = tf
        self._policy = tf.saved_model.load(str(policy_dir))
        self._mean_policy = tf.function(lambda obs: self._policy(obs).mean(), autograph=False)

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "OfficialFlightPolicy":
        return cls(ensure_official_flight_policy(cache_dir))

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        batched = {
            key: self._tf.convert_to_tensor(np.expand_dims(value, 0), dtype=self._tf.float32)
            for key, value in observation.items()
        }
        action = self._mean_policy(batched)[0].numpy().astype(np.float32, copy=False)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"Official flight policy returned invalid action shape/value: {action.shape}"
            )
        return action
