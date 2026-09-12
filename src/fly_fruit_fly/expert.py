"""Loader for the official Flybody flight policy released by HHMI Janelia/DeepMind."""

from __future__ import annotations

import hashlib
import io
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

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "OfficialFlightPolicy":
        return cls(ensure_official_flight_policy(cache_dir))

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        batched = {
            key: self._tf.convert_to_tensor(np.expand_dims(value, 0), dtype=self._tf.float32)
            for key, value in observation.items()
        }
        distribution = self._policy(batched)
        action = distribution.mean()[0].numpy().astype(np.float32, copy=False)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"Official flight policy returned invalid action shape/value: {action.shape}"
            )
        return action
