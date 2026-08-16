"""Pure-Python validation and evidence helpers for ELF3 HIM runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CANONICAL_NUM_ENVS = 4096
CANONICAL_ITERATIONS = 2000
CANONICAL_STEPS_PER_ENV = 50
ORDINARY_PLAY_STEPS = 100
EVALUATION_STEPS = 1000
EVALUATION_NUM_ENVS = 16
_DEVICE = re.compile(r"(?:cpu|cuda:[0-9]+)\Z")
_SCALARS = frozenset(
    {"reward", "policy_loss", "value_loss", "estimator_loss",
     "learning_rate", "entropy", "throughput"}
)
_LIMITS = {
    "stand": {"survival": 0.95, "height_mae": 0.08, "tilt_rms": 0.20},
    "forward": {"survival": 0.90, "velocity_mae": 0.20, "height_mae": 0.10},
    "turn": {"survival": 0.90, "yaw_rate_mae": 0.25, "height_mae": 0.10},
    "crouch": {"survival": 0.90, "height_mae": 0.08, "planar_speed_rms": 0.15},
}


@dataclass(frozen=True)
class EvaluationScenario:
    command: tuple[float, float, float, float | None]
    mode: int


EVALUATION_SCENARIOS = {
    "stand": EvaluationScenario((0.0, 0.0, 0.0, None), 1),
    "forward": EvaluationScenario((0.5, 0.0, 0.0, None), 0),
    "turn": EvaluationScenario((0.0, 0.0, 0.5, None), 0),
    "crouch": EvaluationScenario((0.0, 0.0, 0.0, 0.80), 2),
}


@dataclass(frozen=True)
class RunRequest:
    command: str
    run_dir: Path
    device: str
    seed: int
    num_envs: int
    iterations: int | None = None
    checkpoint: Path | None = None
    resume: bool = False
    headless: bool = False
    scenario: str | None = None
    steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _device(value: str) -> str:
    if _DEVICE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("device must be cpu or cuda:<index>")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True, type=_device)
    parser.add_argument("--seed", required=True, type=_nonnegative)
    parser.add_argument("--num-envs", required=True, type=_positive)
    parser.add_argument("--headless", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ELF3 HIM workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--iterations", required=True, type=_positive)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--checkpoint")
    play = commands.add_parser("play")
    _common(play)
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--scenario", choices=tuple(EVALUATION_SCENARIOS))
    play.add_argument("--steps", type=_positive)
    export = commands.add_parser("export")
    _common(export)
    export.add_argument("--checkpoint", required=True)
    return parser


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    candidate = Path(raw).expanduser()
    if candidate.name.casefold() == "latest" or any(c in raw for c in "*?[]"):
        raise ValueError("checkpoint must be one explicit regular file")
    if not _exists(candidate) or candidate.is_symlink():
        raise ValueError("checkpoint must be an existing non-symlink file")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ValueError("checkpoint cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("checkpoint must be a regular file")
    return candidate.resolve(strict=True)


def _new_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if _exists(candidate):
        raise FileExistsError(f"run path already exists: {candidate}")
    resolved = candidate.parent.resolve(strict=True) / candidate.name
    if _exists(resolved):
        raise FileExistsError(f"run path aliases an existing path: {resolved}")
    return resolved


def parse_request(argv: Sequence[str] | None = None) -> RunRequest:
    args = build_parser().parse_args(argv)
    run_dir = _new_path(args.run_dir)
    resume = bool(getattr(args, "resume", False))
    raw_checkpoint = getattr(args, "checkpoint", None)
    if args.command == "train":
        if resume and raw_checkpoint is None:
            raise ValueError("resume requires --checkpoint")
        if not resume and raw_checkpoint is not None:
            raise ValueError("fresh training rejects --checkpoint")
    scenario = getattr(args, "scenario", None)
    requested_steps = getattr(args, "steps", None)
    if args.command == "play":
        if (scenario is None) != (requested_steps is None):
            raise ValueError("--scenario and --steps must be provided together")
        if scenario is None:
            steps = ORDINARY_PLAY_STEPS
        else:
            if requested_steps != EVALUATION_STEPS:
                raise ValueError("scenario play requires exactly 1000 steps")
            if args.num_envs != EVALUATION_NUM_ENVS:
                raise ValueError("scenario play requires exactly 16 environments")
            steps = EVALUATION_STEPS
    else:
        steps = None
    checkpoint = resolve_checkpoint(raw_checkpoint) if raw_checkpoint else None
    return RunRequest(
        command=args.command,
        run_dir=run_dir,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        iterations=getattr(args, "iterations", None),
        checkpoint=checkpoint,
        resume=resume,
        headless=args.headless,
        scenario=scenario,
        steps=steps,
    )


def create_run_directory(path: str | os.PathLike[str]) -> Path:
    resolved = _new_path(path)
    try:
        resolved.mkdir(mode=0o755)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ValueError(f"cannot create run directory: {resolved}") from exc
    return resolved


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON evidence cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON evidence keys must be strings")
        return {key: _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def write_json_once(path: str | os.PathLike[str], payload: Any) -> Path:
    target = Path(path)
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    if _exists(target):
        raise FileExistsError(f"evidence already exists: {target}")
    temporary = None
    reservation = None
    reservation_identity = None
    replaced = False
    try:
        reservation = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        reserved_stat = os.fstat(reservation)
        reservation_identity = (reserved_stat.st_dev, reserved_stat.st_ino)
        os.close(reservation)
        reservation = None
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        replaced = True
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return target
    finally:
        if reservation is not None:
            os.close(reservation)
        if temporary is not None and _exists(temporary):
            temporary.unlink()
        if not replaced and reservation_identity is not None:
            try:
                target_stat = target.lstat()
            except FileNotFoundError:
                pass
            else:
                target_identity = (target_stat.st_dev, target_stat.st_ino)
                if target_identity == reservation_identity:
                    target.unlink()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tensorboard_scalars(
    log_dir: str | os.PathLike[str],
    required_tags: Sequence[str],
    minimum_points: int,
) -> dict[str, list[float]]:
    if (
        isinstance(minimum_points, bool)
        or not isinstance(minimum_points, int)
        or minimum_points <= 0
    ):
        raise ValueError("minimum_points must be a positive integer")
    if isinstance(required_tags, (str, bytes)) or not required_tags:
        raise ValueError("required_tags must be a nonempty sequence")
    if any(not isinstance(tag, str) or not tag for tag in required_tags):
        raise TypeError("TensorBoard tags must be nonempty strings")
    if len(set(required_tags)) != len(required_tags):
        raise ValueError("TensorBoard tags must be unique")

    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    accumulator = EventAccumulator(os.fspath(log_dir))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    result: dict[str, list[float]] = {}
    for tag in required_tags:
        if tag not in available:
            raise KeyError(f"missing TensorBoard scalar: {tag}")
        events = sorted(
            accumulator.Scalars(tag),
            key=lambda event: (event.step, event.wall_time),
        )
        if len(events) < minimum_points:
            raise ValueError(
                f"TensorBoard scalar {tag} has insufficient points"
            )
        result[tag] = [
            _number(event.value, f"TensorBoard scalar {tag}")
            for event in events
        ]
    return result


def classify_run(path: str | os.PathLike[str]) -> str:
    run = Path(path)
    if not _exists(run):
        return "ABSENT"
    if run.is_symlink() or not run.is_dir():
        raise ValueError("run path must be a non-symlink directory")
    if not (run / "manifest.json").is_file():
        return "ABSENT"
    result_path = run / "result.json"
    if not result_path.is_file():
        return "INCOMPLETE"
    status = json.loads(result_path.read_text(encoding="utf-8")).get("status")
    if status not in {"PASS", "FAIL"}:
        raise ValueError("result status must be PASS or FAIL")
    return status


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_checkpoint_payload(
    payload: Any, *, require_optimizers: bool
) -> int:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    schema = payload.get("schema_version")
    if isinstance(schema, bool) or schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if "model_state_dict" not in payload:
        raise KeyError("model_state_dict")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise TypeError("model_state_dict must be a mapping")
    optimizer_names = (
        "optimizer_state_dict",
        "estimator_optimizer_state_dict",
    )
    for name in optimizer_names:
        if name in payload and not isinstance(payload[name], Mapping):
            raise TypeError(f"{name} must be a mapping")
    for name in ("learning_rate", "estimator_learning_rate"):
        _number(payload[name], name)
    if not isinstance(payload.get("estimator_lr_follows_policy"), bool):
        raise TypeError("estimator_lr_follows_policy must be boolean")
    iteration = payload.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iter must be an integer")
    if iteration < 0:
        raise ValueError("iter must be nonnegative")
    if require_optimizers:
        for name in optimizer_names:
            if name not in payload:
                raise KeyError(name)
    return iteration


def final_iteration(start: int, additional: int) -> int:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if (
        isinstance(additional, bool)
        or not isinstance(additional, int)
        or additional <= 0
    ):
        raise ValueError("additional must be a positive integer")
    return start + additional


def canonical_iterations(
    num_envs: int, num_steps_per_env: int = CANONICAL_STEPS_PER_ENV
) -> int:
    for value, name in (
        (num_envs, "num_envs"),
        (num_steps_per_env, "num_steps_per_env"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    budget = (
        CANONICAL_NUM_ENVS
        * CANONICAL_ITERATIONS
        * CANONICAL_STEPS_PER_ENV
    )
    return math.ceil(budget / (num_envs * num_steps_per_env))


def _require_relative(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("repository paths must be strings")
    if (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError("manifest paths must be repository-relative")


def validate_manifest(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("manifest must be a mapping")
    required = {
        "schema_version", "command", "created_utc", "task_id", "seed",
        "device", "num_envs", "iterations", "cli", "git", "configs",
        "assets", "m4_sources", "runtime", "gpu",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"manifest missing fields: {sorted(missing)}")
    schema = payload["schema_version"]
    if isinstance(schema, bool) or schema != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    if (
        payload["command"] not in {"train", "play", "export"}
        or payload["task_id"] != TASK_ID
    ):
        raise ValueError("invalid manifest identity")
    _json_safe(payload)
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] < 0
    ):
        raise ValueError("manifest seed must be a nonnegative integer")
    value = payload["num_envs"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("manifest num_envs must be a positive integer")
    iterations = payload["iterations"]
    if payload["command"] == "train":
        value = iterations
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("train manifest iterations must be a positive integer")
    elif iterations is not None:
        raise ValueError("play/export manifest iterations must be null")
    if _DEVICE.fullmatch(payload["device"]) is None:
        raise ValueError("invalid manifest device")
    for name in ("git", "configs", "assets", "m4_sources", "runtime", "gpu"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"manifest {name} must be a mapping")
    dirty_paths = payload["git"].get("dirty_paths")
    if not isinstance(dirty_paths, list):
        raise TypeError("git.dirty_paths must be a list")
    for path in dirty_paths:
        _require_relative(path)
    source_files = payload["m4_sources"].get("files", {})
    if not isinstance(source_files, Mapping):
        raise TypeError("m4_sources.files must be a mapping")
    for path in source_files:
        _require_relative(path)
    for name in ("total_mib", "free_mib"):
        if _number(payload["gpu"][name], f"gpu.{name}") < 0:
            raise ValueError(f"gpu.{name} must be nonnegative")


def _series(value: Any, name: str, minimum: int = 1) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise ValueError(f"{name} has insufficient points")
    return [_number(point, name) for point in value]


def evaluate_convergence(payload: Any) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        if not isinstance(payload, Mapping):
            raise TypeError("convergence evidence must be a mapping")
        lengths = _series(
            payload["mean_episode_length"], "mean_episode_length", 200
        )
        timeouts = _series(payload["timeouts"], "timeouts", 100)
        scalars = payload["scalars"]
        if not isinstance(scalars, Mapping):
            raise TypeError("scalars must be a mapping")
        missing = _SCALARS.difference(scalars)
        if missing:
            raise KeyError(f"missing scalars: {sorted(missing)}")
        for name in _SCALARS:
            _series(scalars[name], name, 200)
        first_mean = sum(lengths[:100]) / 100
        last_mean = sum(lengths[-100:]) / 100
        if last_mean <= 300:
            reasons.append("final mean must exceed 300")
        if last_mean < 4 * first_mean:
            reasons.append("final mean must be at least four times the first")
        if sum(point > 0 for point in timeouts[-100:]) < 5:
            reasons.append("final timeout window requires five positive points")
        actual = _number(payload["actual_transitions"], "actual_transitions")
        expected = _number(
            payload["expected_transitions"], "expected_transitions"
        )
        if actual != expected:
            reasons.append("transition budget is incomplete")
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(str(exc))
    return {"passed": not reasons, "reasons": reasons}


def evaluate_behavior(payload: Any) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("rows"), list)
        ):
            raise TypeError("behavior evidence must contain rows")
        expected = {
            (seed, scenario)
            for seed in (42, 43, 44)
            for scenario in _LIMITS
        }
        indexed = {}
        for row in payload["rows"]:
            if not isinstance(row, Mapping):
                raise TypeError("behavior rows must be mappings")
            key = (row.get("seed"), row.get("scenario"))
            if key in indexed:
                raise ValueError(f"duplicate behavior row: {key}")
            indexed[key] = row
        if set(indexed) != expected:
            raise ValueError("behavior rows must cover all seeds and scenarios")
        for (seed, scenario), row in indexed.items():
            if row.get("finite") is not True:
                reasons.append(f"{scenario} seed {seed} is non-finite")
            survival = _number(row["survival"], f"{scenario}.survival")
            if survival < _LIMITS[scenario]["survival"]:
                reasons.append(f"{scenario} seed {seed} survival is too low")
        for scenario, limits in _LIMITS.items():
            rows = [indexed[(seed, scenario)] for seed in (42, 43, 44)]
            for metric, limit in limits.items():
                if metric == "survival":
                    continue
                mean_value = sum(
                    _number(row[metric], f"{scenario}.{metric}")
                    for row in rows
                ) / len(rows)
                if mean_value > limit:
                    reasons.append(f"{scenario} {metric} exceeds {limit}")
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(str(exc))
    return {"passed": not reasons, "reasons": reasons}
