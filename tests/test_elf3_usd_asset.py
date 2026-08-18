"""The committed ELF3 USD must exist, be current, and be self-contained.

These checks do not need Isaac Sim: they compare the conversion stamp against
the on-disk URDF and meshes. The articulation itself is validated by
`isaaclab_ext/scripts/check_elf3_asset.py`, which does require Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from openhomie_isaaclab import elf3_constants as C


STAMP_NAME = "elf3.usd.stamp.json"


def _digest(path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


@pytest.fixture(scope="module")
def stamp() -> dict[str, str]:
    stamp_path = C.USD_PATH.parent / STAMP_NAME
    if not stamp_path.is_file():
        pytest.skip(
            "no conversion stamp; run isaaclab_ext/scripts/convert_elf3_usd.py"
        )
    return json.loads(stamp_path.read_text())


def test_usd_exists():
    if not C.USD_PATH.is_file():
        pytest.skip("run isaaclab_ext/scripts/convert_elf3_usd.py first")
    assert C.USD_PATH.stat().st_size > 0


def test_usd_is_not_stale(stamp):
    """A URDF or mesh edit must invalidate the committed USD.

    The previous attempt converted on every launch, which was slow and filled
    /tmp with 2.1 GB of duplicates. Converting once means staleness has to be
    detected explicitly instead.
    """
    assert stamp.get("elf3.urdf") == _digest(C.URDF_PATH), (
        "elf3.usd was built from a different URDF; rerun convert_elf3_usd.py"
    )


def test_every_mesh_is_covered_by_the_stamp(stamp):
    on_disk = {f"meshes/{p.name}" for p in (C.ASSET_ROOT / "meshes").glob("*.STL")}
    recorded = {key for key in stamp if key.startswith("meshes/")}
    assert recorded == on_disk


def test_mesh_digests_are_current(stamp):
    for key, digest in stamp.items():
        if not key.startswith("meshes/"):
            continue
        path = C.ASSET_ROOT / key
        assert path.is_file(), key
        assert _digest(path) == digest, f"{key} changed; rerun convert_elf3_usd.py"


def test_usd_payload_is_committed_alongside_the_stage():
    """The stage references configuration/*.usd, so those must travel with it."""
    if not C.USD_PATH.is_file():
        pytest.skip("run isaaclab_ext/scripts/convert_elf3_usd.py first")
    configuration = C.USD_PATH.parent / "configuration"
    assert configuration.is_dir(), "missing USD payload directory"
    produced = {p.name for p in configuration.glob("*.usd")}
    assert "elf3_base.usd" in produced
    assert "elf3_physics.usd" in produced


def test_usd_path_is_inside_the_extension():
    """Guards against a converter writing to a temp dir or absolute path."""
    assert C.USD_PATH.parent == C.ASSET_ROOT
    assert not str(C.USD_PATH).startswith("/tmp")
