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

    Only the ``flight/`` subtree is extracted. The upstream archive also contains
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


def _load_tensorflow_runtime():
    """Import TensorFlow/TFP and eagerly register legacy distribution TypeSpecs.

    The released SavedModel was produced with TFP 0.16. Newer TFP releases are
    lazy-loaded, so importing only ``tensorflow_probability`` is not sufficient
    to register the serialized ``Independent_ACTTypeSpec`` / ``Normal_ACTTypeSpec``
    names before ``tf.saved_model.load`` decodes the object graph.
    """
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
        # Import concrete distribution modules, not only the lazy TFP facade.
        # These imports execute AutoCompositeTensor registration before loading
        # the legacy SavedModel object graph.
        from tensorflow_probability.python.distributions import independent  # noqa: F401
        from tensorflow_probability.python.distributions import normal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Official expert evaluation needs the optional TensorFlow runtime. "
            "Install with: pip install -e '.[expert]'"
        ) from exc
    return tf, tfp


class OfficialFlightPolicy:
    """Deterministic wrapper around the released TensorFlow SavedModel."""

    def __init__(self, policy_dir: Path):
        tf, self._tfp = _load_tensorflow_runtime()
        self._tf = tf
        try:
            self._policy = tf.saved_model.load(str(policy_dir))
        except ValueError as exc:
            raise RuntimeError(
                "Could not deserialize the released Flybody policy with the installed "
                "TensorFlow Probability runtime. Install the pinned expert extra "
                "(`pip install -e '.[expert]'`) and retry."
            ) from exc

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "OfficialFlightPolicy":
        return cls(ensure_official_flight_policy(cache_dir))

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        batched = {
            key: self._tf.convert_to_tensor(np.expand_dims(value, 0))
            for key, value in observation.items()
        }
        distribution = self._policy(batched)
        action = distribution.mean()[0].numpy().astype(np.float32, copy=False)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"Official flight policy returned invalid action shape/value: {action.shape}"
            )
        return action
