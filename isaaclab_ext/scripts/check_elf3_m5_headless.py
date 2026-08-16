#!/usr/bin/env python3
"""Run and verify the fixed ELF3 M5 Batch B headless acceptance sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

NUM_ENVS = 16
SEED = 42
TRAIN_ITERATIONS = 2
RESUME_ITERATIONS = 1
PLAY_STEPS = 100
EXPORT_NUM_ENVS = 1
MIN_FREE_GPU_MIB = 4096
TS_MAX_ERROR = 1e-7
ONNX_MAX_ERROR = 1e-5

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    REPO_ROOT
    / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
)
URDF_PATH = PACKAGE_ROOT / "assets/elf3/elf3.urdf"
USD_PATH = PACKAGE_ROOT / "assets/elf3/elf3.usd"
M4_PATHS = (
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/__init__.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/actor_critic.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/estimator.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/exporter.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/ppo.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/storage.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/symmetry.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/__init__.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/him_ppo_cfg.py",
)
EXPECTED_VERSIONS = {
    "isaaclab": "0.54.2",
    "isaaclab-rl": "0.4.7",
    "rsl-rl-lib": "3.1.2",
    "torch": "2.7.0+cu128",
    "onnx": "1.21.0",
    "onnxruntime": "1.28.0",
}

TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
APP_CLOSE_MARKER = "M5_APP_CLOSE_RETURNED"
CHILD_NAMES = ("train", "resume", "play", "export")
FINITE_TRAINING_KEYS = frozenset(
    {
        "observations",
        "actions",
        "rewards",
        "losses",
        "learning_rates",
        "entropy",
        "estimator_metrics",
        "checkpoint_values",
    }
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _walk_finite(value: Any, name: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string key")
            _walk_finite(nested, f"{name}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_finite(nested, f"{name}[{index}]")


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load strict finite JSON from one regular, non-symlink file."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"JSON evidence is not a regular file: {source}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON evidence contains {value}")

    payload = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    _walk_finite(payload)
    return payload


def gpu_preflight(device: str = "cuda:0") -> dict[str, Any]:
    """Execute a CUDA tensor operation and check Blackwell memory evidence."""
    if device != "cuda:0":
        raise ValueError("the Batch B harness accepts only cuda:0")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability < (12, 0):
        raise RuntimeError(f"CUDA capability {capability} is below sm_120")
    arch_list = tuple(str(value) for value in torch.cuda.get_arch_list())
    if "sm_120" not in arch_list:
        raise RuntimeError("the installed Torch build does not contain sm_120")
    probe = torch.ones(32, device=device).square().sum().item()
    if not math.isfinite(float(probe)) or float(probe) != 32.0:
        raise RuntimeError("CUDA tensor execution returned invalid data")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    selected: tuple[str, str, int, int] | None = None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5 and fields[0] == "0":
            selected = (fields[1], fields[2], int(fields[3]), int(fields[4]))
            break
    if selected is None:
        raise RuntimeError("nvidia-smi did not report GPU index 0")
    name, driver_version, total_mib, free_mib = selected
    if free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(
            f"cuda:0 has {free_mib} MiB free; {MIN_FREE_GPU_MIB} MiB required"
        )
    return {
        "device": device,
        "name": name,
        "driver_version": driver_version,
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "")),
        "capability": list(capability),
        "arch_list": list(arch_list),
        "total_mib": total_mib,
        "free_mib": free_mib,
        "cuda_probe": float(probe),
        "cuda_probe_passed": True,
    }


def run_child(
    command: Sequence[str], log_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Run one child without a shell and capture its complete close boundary."""
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("child command must be a nonempty sequence")
    argv = [str(value) for value in command]
    if not Path(argv[0]).is_absolute():
        raise ValueError("child interpreter path must be absolute")
    target = Path(log_path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"child log already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    if output and not output.endswith("\n"):
        output += "\n"
    output += APP_CLOSE_MARKER + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(output)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "returncode": completed.returncode,
        "command": argv,
        "log_path": str(target.resolve()),
    }


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing {name}: {path}")
    return path.resolve(strict=True)


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def _valid_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _check_log(record: Mapping[str, Any], expected: str) -> None:
    if record.get("returncode") != 0:
        raise ValueError(f"{expected} child did not exit zero")
    log_path = _absolute_path(record.get("log_path"), f"{expected} log")
    text = _regular_file(log_path, f"{expected} log").read_text(
        encoding="utf-8"
    )
    pass_matches = re.findall(r"(?m)^M5_HEADLESS_PASS\s*$", text)
    internal_matches = re.findall(r"(?m)^M5_INTERNAL_EXIT_CODE=0\s*$", text)
    if len(pass_matches) != 1 or len(internal_matches) != 1:
        raise ValueError(f"{expected} log needs exactly one PASS and internal zero")
    if "M5_HEADLESS_FAIL" in text or "M5_INTERNAL_EXIT_CODE=1" in text:
        raise ValueError(f"{expected} log contains a failure sentinel")
    if re.search(r"\b(?:skip(?:ped|s)?|xfail(?:ed)?)\b", text, re.IGNORECASE):
        raise ValueError(f"{expected} log contains a skip/xfail marker")
    if text.count(APP_CLOSE_MARKER) != 1:
        raise ValueError(f"{expected} log needs one application-close marker")
    if text.index("M5_HEADLESS_PASS") > text.index(APP_CLOSE_MARKER):
        raise ValueError(f"{expected} PASS sentinel is after application close")


def _check_manifest(
    payload: Any,
    *,
    command: str,
    num_envs: int,
    iterations: int | None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("manifest must be a mapping")
    expected = {
        "schema_version": 1,
        "command": command,
        "task_id": TASK_ID,
        "seed": SEED,
        "device": "cuda:0",
        "num_envs": num_envs,
        "iterations": iterations,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"manifest {key} does not match the Batch B contract")
    _check_identity_evidence(payload)
    return payload


def _check_result(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        raise ValueError(f"{name} result is not PASS")
    return payload


def _check_identity_evidence(payload: Mapping[str, Any]) -> None:
    git = payload.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("manifest git identity is missing")
    if re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", ""))) is None:
        raise ValueError("manifest git commit is not a full lowercase hash")
    dirty_paths = git.get("dirty_paths")
    if not isinstance(dirty_paths, list):
        raise ValueError("manifest git dirty paths are missing")
    for path in dirty_paths:
        if (
            not isinstance(path, str)
            or Path(path).is_absolute()
            or PureWindowsPath(path).is_absolute()
        ):
            raise ValueError("manifest dirty paths must be repository-relative")

    configs = payload.get("configs")
    if not isinstance(configs, Mapping):
        raise ValueError("manifest config evidence is missing")
    hashes = configs.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("manifest config hashes are missing")
    for name in ("env", "agent"):
        value = configs.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"manifest {name} config is missing")
        if hashes.get(name) != _json_sha256(value):
            raise ValueError(f"manifest {name} config hash does not match")

    assets = payload.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("manifest asset evidence is missing")
    expected_assets = {
        "urdf_sha256": _sha256(_regular_file(URDF_PATH, "ELF3 URDF")),
        "usd_sha256": _sha256(_regular_file(USD_PATH, "ELF3 USD")),
    }
    if any(assets.get(name) != digest for name, digest in expected_assets.items()):
        raise ValueError("manifest asset hashes do not match the accepted assets")

    m4_sources = payload.get("m4_sources")
    if not isinstance(m4_sources, Mapping):
        raise ValueError("manifest M4 source evidence is missing")
    expected_files = {
        path: _sha256(_regular_file(REPO_ROOT / path, f"M4 source {path}"))
        for path in M4_PATHS
    }
    if m4_sources.get("files") != expected_files:
        raise ValueError("manifest M4 file hashes do not match the accepted stack")
    if m4_sources.get("sha256") != _json_sha256(expected_files):
        raise ValueError("manifest aggregate M4 hash does not match")

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("manifest runtime evidence is missing")
    if not isinstance(runtime.get("python"), str) or not runtime["python"]:
        raise ValueError("manifest Python version is missing")
    if runtime.get("versions") != EXPECTED_VERSIONS:
        raise ValueError("manifest dependency versions do not match")
    for name in ("isaaclab_path", "isaaclab_app_path", "isaaclab_rl_path"):
        value = runtime.get(name)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or "IsaacLab-v2.3.2" not in Path(value).parts
        ):
            raise ValueError(f"manifest {name} is outside IsaacLab-v2.3.2")

    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ValueError("manifest GPU evidence is missing")
    for name in ("name", "driver_version", "cuda_version"):
        if not isinstance(gpu.get(name), str) or not gpu[name]:
            raise ValueError(f"manifest GPU {name} is missing")
    total_mib = _number(gpu.get("total_mib"), "gpu.total_mib")
    free_mib = _number(gpu.get("free_mib"), "gpu.free_mib")
    if total_mib <= 0 or free_mib < MIN_FREE_GPU_MIB or free_mib > total_mib:
        raise ValueError("manifest GPU memory evidence is invalid")
    capability = gpu.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in capability
        )
        or tuple(capability) < (12, 0)
    ):
        raise ValueError("manifest GPU capability is below sm_120")
    arch_list = gpu.get("arch_list")
    if not isinstance(arch_list, list) or "sm_120" not in arch_list:
        raise ValueError("manifest Torch architectures do not include sm_120")
    if gpu.get("cuda_probe_passed") is not True:
        raise ValueError("manifest CUDA tensor probe did not pass")


def _checkpoint(
    payload: Mapping[str, Any], name: str, expected_iteration: int
) -> tuple[Path, str]:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{name} checkpoint identity is missing")
    path = _absolute_path(checkpoint.get("path"), f"{name} checkpoint")
    path = _regular_file(path, f"{name} checkpoint")
    digest = _valid_hash(checkpoint.get("sha256"), f"{name} checkpoint hash")
    if checkpoint.get("iteration") != expected_iteration:
        raise ValueError(f"{name} checkpoint iteration is wrong")
    if _sha256(path) != digest:
        raise ValueError(f"{name} checkpoint hash does not match")
    return path, digest


def _check_finite_training(result: Mapping[str, Any], name: str) -> None:
    finite = result.get("finite")
    if not isinstance(finite, Mapping) or not FINITE_TRAINING_KEYS.issubset(finite):
        raise ValueError(f"{name} finite training evidence is incomplete")
    if any(finite[key] is not True for key in FINITE_TRAINING_KEYS):
        raise ValueError(f"{name} contains non-finite runtime evidence")


def _expected_commands(root: Path, python: str, script: str, train_ckpt: str, resume_ckpt: str) -> dict[str, list[str]]:
    common = ["--headless", "--device", "cuda:0", "--seed", str(SEED)]
    return {
        "train": [
            python, script, "train", *common, "--num-envs", str(NUM_ENVS),
            "--iterations", str(TRAIN_ITERATIONS), "--run-dir", str(root / "train"),
        ],
        "resume": [
            python, script, "train", "--resume", "--checkpoint", train_ckpt,
            *common, "--num-envs", str(NUM_ENVS), "--iterations",
            str(RESUME_ITERATIONS), "--run-dir", str(root / "resume"),
        ],
        "play": [
            python, script, "play", "--checkpoint", resume_ckpt, *common,
            "--num-envs", str(NUM_ENVS), "--run-dir", str(root / "play"),
        ],
        "export": [
            python, script, "export", "--checkpoint", resume_ckpt, *common,
            "--num-envs", str(EXPORT_NUM_ENVS), "--run-dir", str(root / "export"),
        ],
    }


def verify_headless_evidence(run_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify a complete offline copy of the four-run Batch B evidence tree."""
    root = Path(run_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("run root must be an absolute non-symlink directory")
    children = load_json(root / "children.json")
    if not isinstance(children, list) or len(children) != len(CHILD_NAMES):
        raise ValueError("children.json must contain exactly four child records")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in children:
        if not isinstance(record, Mapping) or record.get("name") not in CHILD_NAMES:
            raise ValueError("invalid child record")
        name = str(record["name"])
        if name in indexed:
            raise ValueError(f"duplicate child record: {name}")
        indexed[name] = record
    if tuple(record.get("name") for record in children) != CHILD_NAMES:
        raise ValueError("children must run in train/resume/play/export order")
    for name in CHILD_NAMES:
        _check_log(indexed[name], name)

    manifests: dict[str, Mapping[str, Any]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    specs = {
        "train": ("train", NUM_ENVS, TRAIN_ITERATIONS),
        "resume": ("train", NUM_ENVS, RESUME_ITERATIONS),
        "play": ("play", NUM_ENVS, None),
        "export": ("export", EXPORT_NUM_ENVS, None),
    }
    for name, (command, num_envs, iterations) in specs.items():
        run = root / name
        if run.is_symlink() or not run.is_dir():
            raise ValueError(f"missing {name} run directory")
        manifests[name] = _check_manifest(
            load_json(run / "manifest.json"),
            command=command,
            num_envs=num_envs,
            iterations=iterations,
        )
        results[name] = _check_result(load_json(run / "result.json"), name)

    train_manifest = manifests["train"]
    train_result = results["train"]
    if train_manifest.get("start_iteration") != 0:
        raise ValueError("fresh training must start at iteration zero")
    if train_result.get("start_iteration") != 0 or train_result.get("final_iteration") != 2:
        raise ValueError("fresh training must advance exactly 0 -> 2")
    _check_finite_training(train_result, "train")
    train_ckpt, train_hash = _checkpoint(train_result, "train", 2)

    resume_manifest = manifests["resume"]
    resume_result = results["resume"]
    parent = resume_manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("resume parent evidence is missing")
    expected_parent = {
        "checkpoint_path": str(train_ckpt),
        "checkpoint_sha256": train_hash,
        "manifest_sha256": _sha256(root / "train" / "manifest.json"),
        "iteration": 2,
    }
    for key, value in expected_parent.items():
        if parent.get(key) != value:
            raise ValueError(f"resume parent {key} does not match")
    if resume_manifest.get("start_iteration") != 2:
        raise ValueError("resume manifest must start at iteration two")
    if resume_result.get("start_iteration") != 2 or resume_result.get("final_iteration") != 3:
        raise ValueError("resume must advance exactly 2 -> 3")
    _check_finite_training(resume_result, "resume")
    resume_ckpt, resume_hash = _checkpoint(resume_result, "resume", 3)

    for name in ("play", "export"):
        source = manifests[name].get("checkpoint")
        if not isinstance(source, Mapping):
            raise ValueError(f"{name} manifest checkpoint identity is missing")
        expected = {
            "path": str(resume_ckpt),
            "sha256": resume_hash,
            "iteration": 3,
        }
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} did not use the exact resume checkpoint")

    play = results["play"].get("play")
    if not isinstance(play, Mapping):
        raise ValueError("play evidence is missing")
    if play.get("steps") != PLAY_STEPS or play.get("action_shape") != [NUM_ENVS, 12]:
        raise ValueError("play step count or action shape is wrong")
    if play.get("finite") is not True or play.get("deterministic") is not True:
        raise ValueError("play outputs are non-finite or nondeterministic")
    if _number(play.get("max_abs_diff"), "play.max_abs_diff") != 0.0:
        raise ValueError("play repeated inference is not bitwise deterministic")
    _valid_hash(play.get("action_sha256"), "play action hash")
    for name in ("sha256_before", "sha256_after"):
        if play.get(name) != resume_hash:
            raise ValueError("play mutated its checkpoint")
    if _sha256(resume_ckpt) != resume_hash:
        raise ValueError("resume checkpoint was mutated after training")

    exports = results["export"].get("exports")
    if not isinstance(exports, Mapping):
        raise ValueError("export evidence is missing")
    for kind, filename, limit in (
        ("torchscript", "policy.ts", TS_MAX_ERROR),
        ("onnx", "policy.onnx", ONNX_MAX_ERROR),
    ):
        artifact = exports.get(kind)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"missing {kind} evidence")
        path = _absolute_path(artifact.get("path"), f"{kind} artifact")
        path = _regular_file(path, f"{kind} artifact")
        if path != (root / "export" / filename).resolve():
            raise ValueError(f"{kind} artifact path is not canonical")
        digest = _valid_hash(artifact.get("sha256"), f"{kind} hash")
        if _sha256(path) != digest:
            raise ValueError(f"{kind} artifact hash does not match")
        if artifact.get("batches") != [1, 4] or artifact.get("fresh_runtime") is not True:
            raise ValueError(f"{kind} did not verify batches 1 and 4 in a fresh runtime")
        if _number(artifact.get("max_abs_error"), f"{kind}.max_abs_error") > limit:
            raise ValueError(f"{kind} parity exceeds its fixed threshold")
    onnx = exports["onnx"]
    if onnx.get("checker_passed") is not True:
        raise ValueError("ONNX checker did not pass")
    if onnx.get("providers") != ["CPUExecutionProvider"]:
        raise ValueError("ONNX parity must execute with CPUExecutionProvider")
    if exports["torchscript"].get("provider") != "torch.jit.load":
        raise ValueError("TorchScript parity must use torch.jit.load")

    python = str(Path(sys.executable).resolve())
    script = str((Path(__file__).resolve().parent / "elf3_him.py").resolve())
    expected_commands = _expected_commands(
        root, python, script, str(train_ckpt), str(resume_ckpt)
    )
    for name in CHILD_NAMES:
        command = indexed[name].get("command")
        if command != expected_commands[name]:
            raise ValueError(f"{name} command does not match the frozen harness")
        if "--play-steps" in command:
            raise ValueError("the frozen Batch A CLI has no --play-steps option")
    return {
        "status": "PASS",
        "train_final_iteration": 2,
        "resume_final_iteration": 3,
        "checkpoint_sha256": resume_hash,
        "torchscript_max_abs_error": _number(
            exports["torchscript"]["max_abs_error"], "torchscript.max_abs_error"
        ),
        "onnx_max_abs_error": _number(
            exports["onnx"]["max_abs_error"], "onnx.max_abs_error"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--headless", action="store_true", required=True)
    return parser


def _validate_cli(argv: Sequence[str] | None) -> tuple[Path, str]:
    args = _parser().parse_args(argv)
    root = Path(args.run_root)
    if not root.is_absolute():
        raise ValueError("--run-root must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError("--run-root must not exist")
    if not root.parent.is_dir():
        raise ValueError("--run-root parent must exist")
    if args.device != "cuda:0":
        raise ValueError("--device must be cuda:0")
    if args.headless is not True:
        raise ValueError("--headless is required")
    return root, args.device


def _write_children(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _child_passed(record: Mapping[str, Any], name: str) -> None:
    _check_log(record, name)


def main(argv: Sequence[str] | None = None) -> int:
    exit_code = 1
    try:
        root, device = _validate_cli(argv)
        gpu_preflight(device)
        root.mkdir(mode=0o755)
        python = str(Path(sys.executable).resolve())
        script = str((Path(__file__).resolve().parent / "elf3_him.py").resolve())
        common = ["--headless", "--device", device, "--seed", str(SEED)]
        records: list[dict[str, Any]] = []

        train_command = [
            python, script, "train", *common, "--num-envs", str(NUM_ENVS),
            "--iterations", str(TRAIN_ITERATIONS), "--run-dir", str(root / "train"),
        ]
        train_record = run_child(train_command, root / "logs" / "train.log")
        train_record["name"] = "train"
        records.append(train_record)
        _child_passed(train_record, "train")
        train_result = _check_result(load_json(root / "train" / "result.json"), "train")
        train_ckpt, _ = _checkpoint(train_result, "train", 2)

        resume_command = [
            python, script, "train", "--resume", "--checkpoint", str(train_ckpt),
            *common, "--num-envs", str(NUM_ENVS), "--iterations",
            str(RESUME_ITERATIONS), "--run-dir", str(root / "resume"),
        ]
        resume_record = run_child(resume_command, root / "logs" / "resume.log")
        resume_record["name"] = "resume"
        records.append(resume_record)
        _child_passed(resume_record, "resume")
        resume_result = _check_result(load_json(root / "resume" / "result.json"), "resume")
        resume_ckpt, _ = _checkpoint(resume_result, "resume", 3)

        for name, num_envs in (("play", NUM_ENVS), ("export", EXPORT_NUM_ENVS)):
            command = [
                python, script, name, "--checkpoint", str(resume_ckpt), *common,
                "--num-envs", str(num_envs), "--run-dir", str(root / name),
            ]
            record = run_child(command, root / "logs" / f"{name}.log")
            record["name"] = name
            records.append(record)
            _child_passed(record, name)

        _write_children(root / "children.json", records)
        evidence = verify_headless_evidence(root)
        print(json.dumps(evidence, sort_keys=True, allow_nan=False), flush=True)
        exit_code = 0
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sentinel = "M5_HEADLESS_PASS" if exit_code == 0 else "M5_HEADLESS_FAIL"
    print(sentinel, flush=True)
    print(f"M5_INTERNAL_EXIT_CODE={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
