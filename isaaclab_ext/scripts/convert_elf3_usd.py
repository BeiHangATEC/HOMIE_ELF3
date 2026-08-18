"""Convert the ELF3 URDF to USD once, offline.

The previous attempt spawned the robot straight from `UrdfFileCfg`, so every
launch re-converted 34 STL meshes into 17 convex hulls and left the result in
a fresh temporary directory: 91 cached copies totalling 2.1 GB. Converting
once and committing to `UsdFileCfg` removes that entirely.

The URDF stays the source of record. This script stamps the URDF's hash into
a sidecar so a stale USD is detected rather than silently reused.

    python isaaclab_ext/scripts/convert_elf3_usd.py
    python isaaclab_ext/scripts/convert_elf3_usd.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = REPO_ROOT / "isaaclab_ext" / "source" / "openhomie_isaaclab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

STAMP_NAME = "elf3.usd.stamp.json"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _asset_fingerprint(urdf_path: Path) -> dict[str, str]:
    """Hash the URDF and every mesh, so a mesh edit also invalidates the USD."""
    fingerprint = {"elf3.urdf": _digest(urdf_path)}
    mesh_dir = urdf_path.parent / "meshes"
    for mesh in sorted(mesh_dir.glob("*.STL")):
        fingerprint[f"meshes/{mesh.name}"] = _digest(mesh)
    return fingerprint


def _read_stamp(stamp_path: Path) -> dict[str, str] | None:
    if not stamp_path.is_file():
        return None
    try:
        return json.loads(stamp_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="reconvert even when the existing USD is up to date",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report status and exit non-zero if the USD is missing or stale",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from openhomie_isaaclab import elf3_constants as C

    if not C.URDF_PATH.is_file():
        print(f"error: URDF not found at {C.URDF_PATH}", file=sys.stderr)
        return 1

    stamp_path = C.USD_PATH.parent / STAMP_NAME
    fingerprint = _asset_fingerprint(C.URDF_PATH)
    recorded = _read_stamp(stamp_path)
    up_to_date = C.USD_PATH.exists() and recorded == fingerprint

    if args.check:
        if up_to_date:
            print(f"up to date: {C.USD_PATH}")
            return 0
        reason = "missing" if not C.USD_PATH.exists() else "stale"
        print(f"{reason}: {C.USD_PATH}", file=sys.stderr)
        return 1

    if up_to_date and not args.force:
        print(f"up to date, nothing to do: {C.USD_PATH}")
        return 0

    # Isaac Sim must start before anything imports its USD bindings.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        # Convert into a staging directory so a crash cannot leave a partial
        # USD behind that the stamp would later vouch for.
        staging = C.USD_PATH.parent / ".usd_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        cfg = UrdfConverterCfg(
            asset_path=str(C.URDF_PATH),
            usd_dir=str(staging),
            usd_file_name=C.USD_PATH.name,
            fix_base=False,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            force_usd_conversion=True,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={**C.UPPER_BODY_STIFFNESS, **C.LEG_STIFFNESS},
                    damping={**C.UPPER_BODY_DAMPING, **C.LEG_DAMPING},
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        produced = Path(converter.usd_path)
        print(f"converted: {produced}")

        # Promote the staged result, replacing any previous conversion.
        if C.USD_PATH.exists():
            C.USD_PATH.unlink()
        legacy_dirs = [
            p for p in C.USD_PATH.parent.glob("configuration") if p.is_dir()
        ]
        for path in legacy_dirs:
            shutil.rmtree(path)
        shutil.move(str(produced), str(C.USD_PATH))
        for extra in staging.iterdir():
            shutil.move(str(extra), str(C.USD_PATH.parent / extra.name))
        shutil.rmtree(staging)

        stamp_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
        print(f"wrote {C.USD_PATH}")
        print(f"wrote {stamp_path}")
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
