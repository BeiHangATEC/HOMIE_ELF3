"""Isolated C3 video capture and evidence helpers for the ELF3 policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from openhomie_isaaclab.workflows.elf3_run import sha256_file, write_json_once


SCENARIO = "forward"
SEED = 42
NUM_ENVS = 16
STEPS = 1000
FPS = 50
WIDTH = 1280
HEIGHT = 720
DURATION_SECONDS = 20.0
DEVICE = "cuda:0"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
SCHEMA_VERSION = 1
C3_WORKFLOW = "elf3_c3_weights_only"
C3_FINAL_STAGE = "V3"
C3_LOCAL_FINAL = 2000
C3_GLOBAL_FINAL = 10000

CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
CAMERA_ORIGIN_TYPE = "asset_root"
CAMERA_ENV_INDEX = 0
CAMERA_ASSET_NAME = "robot"
CAMERA_EYE = (3.2, 2.8, 1.8)
CAMERA_LOOKAT = (0.0, 0.0, 0.55)

SAMPLE_FRAME_INDICES = tuple(range(0, STEPS, 100)) + (STEPS - 1,)
MIN_MEAN_LUMA = 1.0
MIN_NONBLACK_FRACTION = 0.01
NONBLACK_LUMA_THRESHOLD = 8
MIN_MOTION_MEAN_ABS_DIFFERENCE = 1.0
RENDER_WARMUP_ATTEMPTS = 8

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_NAME = re.compile(r"model_([0-9]+)\.pt\Z")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


@dataclass(frozen=True)
class C3VideoRequest:
    checkpoint: Path
    output_root: Path
    plan: Path
    plan_sha256: str
    source_manifest: Path
    source_result: Path
    command: str = "c3_video"
    device: str = DEVICE
    headless: bool = True
    enable_cameras: bool = True
    scenario: str = SCENARIO
    seed: int = SEED
    num_envs: int = NUM_ENVS
    steps: int = STEPS
    fps: int = FPS
    width: int = WIDTH
    height: int = HEIGHT
    iterations: None = None
    resume: bool = False

    @property
    def run_dir(self) -> Path:
        return self.output_root

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in payload.items()
        }


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Record the fixed ELF3 C3 forward acceptance clip"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    return parser


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _regular_file(path: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ValueError(f"{name} cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return candidate.resolve(strict=True)


def _valid_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _checkpoint_iteration(path: Path) -> int:
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("checkpoint filename must be model_<iteration>.pt")
    return int(match.group(1))


def _new_output_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if _exists(candidate):
        raise FileExistsError(f"output root already exists: {candidate}")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("output root parent must already exist") from exc
    resolved = parent / candidate.name
    if not candidate.name or _exists(resolved):
        raise FileExistsError(f"output root already exists: {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def parse_request(argv: Sequence[str] | None = None) -> C3VideoRequest:
    args = build_parser().parse_args(argv)
    checkpoint = _regular_file(args.checkpoint, "checkpoint")
    _checkpoint_iteration(checkpoint)
    source_manifest = _regular_file(
        checkpoint.parent / "manifest.json", "source manifest"
    )
    source_result = _regular_file(
        checkpoint.parent / "result.json", "source result"
    )
    plan = _regular_file(args.plan, "plan")
    plan_sha256 = _valid_sha256(args.plan_sha256, "plan SHA-256")
    if sha256_file(plan) != plan_sha256:
        raise ValueError("plan SHA-256 does not match the plan file")

    output_root = _new_output_root(args.output_root)
    source_run = checkpoint.parent
    if _is_within(output_root, source_run):
        raise ValueError("output root must be outside the source V3 run")

    return C3VideoRequest(
        checkpoint=checkpoint,
        output_root=output_root,
        plan=plan,
        plan_sha256=plan_sha256,
        source_manifest=source_manifest,
        source_result=source_result,
    )


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


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    source = _regular_file(path, name)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    payload = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    _walk_finite(payload, name)
    return payload


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _absolute_recorded_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value).resolve(strict=False)


def _file_reference(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, path.name)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _config_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    configs = manifest.get("configs")
    if not isinstance(configs, Mapping):
        raise TypeError("source manifest configs must be a mapping")
    hashes = configs.get("sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {"env", "agent"}:
        raise ValueError("source manifest config SHA-256 values are incomplete")
    return {
        name: _valid_sha256(hashes[name], f"source {name} config SHA-256")
        for name in ("env", "agent")
    }


def _stage_name(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("name")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("source stage must be a nonempty string")
    return value


def build_source_binding(request: C3VideoRequest) -> dict[str, Any]:
    """Validate and hash-bind the checkpoint, training evidence, and plan."""
    manifest = _load_json(request.source_manifest, "source manifest")
    result = _load_json(request.source_result, "source result")
    if (
        manifest.get("workflow") != C3_WORKFLOW
        or manifest.get("stage") != C3_FINAL_STAGE
        or manifest.get("iterations") != C3_LOCAL_FINAL
    ):
        raise ValueError("source manifest must be the final C3 V3 training stage")
    if (
        result.get("workflow") != C3_WORKFLOW
        or result.get("status") != "PASS"
        or result.get("stage") != C3_FINAL_STAGE
    ):
        raise ValueError("source result must be a passing final C3 V3 result")

    lifecycle = manifest.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise TypeError("source lifecycle must be a mapping")
    local_iterations = lifecycle.get("local_iterations")
    global_iterations = lifecycle.get("global_iterations")
    if (
        not isinstance(local_iterations, Mapping)
        or not isinstance(global_iterations, Mapping)
        or _integer(local_iterations.get("start"), "source local start") != 0
        or _integer(local_iterations.get("final"), "source local final")
        != C3_LOCAL_FINAL
        or _integer(global_iterations.get("start"), "source global start") != 8000
        or _integer(global_iterations.get("final"), "source global final")
        != C3_GLOBAL_FINAL
        or result.get("lifecycle") != lifecycle
    ):
        raise ValueError("source C3 lifecycle is not the required V3 identity")

    manifest_plan = manifest.get("plan")
    if not isinstance(manifest_plan, Mapping):
        raise TypeError("source C3 plan must be a mapping")
    plan_path = _absolute_recorded_path(
        manifest_plan.get("path"), "source C3 plan path"
    )
    plan_hash = _valid_sha256(
        manifest_plan.get("sha256"), "source C3 plan SHA-256"
    )
    if plan_path != request.plan or plan_hash != request.plan_sha256:
        raise ValueError("source C3 training used a different plan")

    checkpoint_hash = sha256_file(request.checkpoint)
    recorded_checkpoint = result.get("checkpoint")
    if not isinstance(recorded_checkpoint, Mapping):
        raise TypeError("source result checkpoint must be a mapping")
    recorded_path = _absolute_recorded_path(
        recorded_checkpoint.get("path"), "source checkpoint path"
    )
    if recorded_path != request.checkpoint:
        raise ValueError("source result references a different checkpoint")
    recorded_hash = _valid_sha256(
        recorded_checkpoint.get("sha256"), "source checkpoint SHA-256"
    )
    if recorded_hash != checkpoint_hash:
        raise ValueError("source result checkpoint SHA-256 does not match")
    iteration = _integer(
        recorded_checkpoint.get("iteration"), "source checkpoint iteration"
    )
    if (
        iteration != _checkpoint_iteration(request.checkpoint)
        or iteration != C3_LOCAL_FINAL
        or recorded_checkpoint.get("stage") != C3_FINAL_STAGE
        or _integer(
            recorded_checkpoint.get("local_iteration"),
            "source checkpoint local iteration",
        )
        != C3_LOCAL_FINAL
        or _integer(
            recorded_checkpoint.get("global_iteration"),
            "source checkpoint global iteration",
        )
        != C3_GLOBAL_FINAL
    ):
        raise ValueError("source checkpoint is not the final C3 V3 identity")
    seed = _integer(manifest.get("seed"), "source training seed")
    num_envs = _integer(
        manifest.get("num_envs"), "source training environment count", minimum=1
    )
    plan_ref = _file_reference(request.plan)
    if plan_ref["sha256"] != request.plan_sha256:
        raise ValueError("plan SHA-256 changed before source binding")
    return {
        "checkpoint": {
            "path": str(request.checkpoint),
            "sha256": checkpoint_hash,
            "iteration": iteration,
        },
        "manifest": _file_reference(request.source_manifest),
        "result": _file_reference(request.source_result),
        "plan": plan_ref,
        "training": {
            "stage": _stage_name(manifest.get("stage")),
            "seed": seed,
            "num_envs": num_envs,
            "local_start_iteration": 0,
            "local_final_iteration": C3_LOCAL_FINAL,
            "global_start_iteration": 8000,
            "global_final_iteration": C3_GLOBAL_FINAL,
            "config_sha256": _config_hashes(manifest),
        },
    }


def verify_source_binding(binding: Mapping[str, Any]) -> None:
    """Fail if any source file has changed or become an alias."""
    for name in ("checkpoint", "manifest", "result", "plan"):
        reference = binding.get(name)
        if not isinstance(reference, Mapping):
            raise RuntimeError(f"{name} binding is missing")
        try:
            expected = _valid_sha256(reference.get("sha256"), f"{name} SHA-256")
            path_value = reference.get("path")
            if not isinstance(path_value, str) or not Path(path_value).is_absolute():
                raise ValueError(f"{name} path must be absolute")
            path = _regular_file(path_value, name)
            if sha256_file(path) != expected:
                raise ValueError(f"{name} SHA-256 changed")
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} source binding changed: {exc}") from exc


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _exact_integer(value: Any, expected: int, name: str) -> int:
    actual = _integer(value, name)
    if actual != expected:
        raise ValueError(f"{name} must equal {expected}")
    return actual


def _relative_video_path(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("video relative_path must be a string")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "\\" in value
        or ".." in posix.parts
        or len(posix.parts) != 2
        or posix.parts[0] != "videos"
        or posix.suffix.casefold() != ".mp4"
    ):
        raise ValueError("video relative_path must name one MP4 in videos/")
    return value


def validate_video_metadata(metadata: Any) -> Mapping[str, Any]:
    """Validate exact C3 MP4 properties and sampled visual evidence."""
    if not isinstance(metadata, Mapping):
        raise TypeError("video metadata must be a mapping")
    _relative_video_path(metadata.get("relative_path"))
    _valid_sha256(metadata.get("sha256"), "video SHA-256")
    size = _integer(metadata.get("size_bytes"), "video size", minimum=1)
    if size <= 0:
        raise ValueError("video size must be positive")
    _exact_integer(metadata.get("frame_count"), STEPS, "frame_count")
    fps = _number(metadata.get("fps"), "fps")
    if not math.isclose(fps, FPS, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"fps must equal {FPS}")
    _exact_integer(metadata.get("width"), WIDTH, "width")
    _exact_integer(metadata.get("height"), HEIGHT, "height")
    duration = _number(metadata.get("duration_seconds"), "duration")
    if not math.isclose(
        duration, DURATION_SECONDS, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(f"duration must equal {DURATION_SECONDS}")

    sampled = metadata.get("sampled_frames")
    if not isinstance(sampled, list) or len(sampled) != len(SAMPLE_FRAME_INDICES):
        raise ValueError("sampled_frames must cover the fixed frame indices")
    actual_indices: list[int] = []
    for position, sample in enumerate(sampled):
        if not isinstance(sample, Mapping):
            raise TypeError("sampled frame evidence must be mappings")
        index = _integer(sample.get("index"), "sample frame index")
        actual_indices.append(index)
        mean_luma = _number(sample.get("mean_luma"), "sample mean_luma")
        nonblack = _number(
            sample.get("nonblack_fraction"), "sample nonblack_fraction"
        )
        if (
            mean_luma < MIN_MEAN_LUMA
            or nonblack < MIN_NONBLACK_FRACTION
            or nonblack > 1.0
        ):
            raise ValueError(f"sampled frame {position} is black")
    if tuple(actual_indices) != SAMPLE_FRAME_INDICES:
        raise ValueError("sampled frame indices are not canonical")

    motion = metadata.get("motion")
    if not isinstance(motion, Mapping):
        raise TypeError("motion evidence must be a mapping")
    pair_count = _exact_integer(
        motion.get("pair_count"), len(SAMPLE_FRAME_INDICES) - 1, "motion pair_count"
    )
    moving = _integer(
        motion.get("moving_pair_count"), "motion moving_pair_count"
    )
    maximum = _number(
        motion.get("max_mean_abs_difference"),
        "motion max_mean_abs_difference",
    )
    if moving < 1 or moving > pair_count or maximum < MIN_MOTION_MEAN_ABS_DIFFERENCE:
        raise ValueError("motion evidence is insufficient")
    return metadata


def inspect_video(video_path: Path, output_root: Path) -> dict[str, Any]:
    """Decode every frame and return validated MP4 evidence."""
    video = _regular_file(video_path, "video MP4")
    root = output_root.resolve(strict=True)
    try:
        relative = video.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("video MP4 escaped the output root") from exc

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("video MP4 cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frames: dict[int, Any] = {}
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tuple(frame.shape[:2]) != (HEIGHT, WIDTH):
                raise ValueError("decoded video frame dimensions changed")
            if frame_count in SAMPLE_FRAME_INDICES:
                frames[frame_count] = frame.copy()
            frame_count += 1
            if frame_count > STEPS:
                break
    finally:
        capture.release()
    if set(frames) != set(SAMPLE_FRAME_INDICES):
        raise ValueError("video MP4 is missing sampled frames")

    sampled_frames: list[dict[str, Any]] = []
    ordered_frames = [frames[index] for index in SAMPLE_FRAME_INDICES]
    for index, frame in zip(SAMPLE_FRAME_INDICES, ordered_frames, strict=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled_frames.append(
            {
                "index": index,
                "mean_luma": float(np.mean(gray)),
                "nonblack_fraction": float(
                    np.count_nonzero(gray > NONBLACK_LUMA_THRESHOLD) / gray.size
                ),
            }
        )
    differences = [
        float(
            np.mean(
                np.abs(
                    current.astype(np.int16, copy=False)
                    - previous.astype(np.int16, copy=False)
                )
            )
        )
        for previous, current in zip(ordered_frames, ordered_frames[1:])
    ]
    metadata = {
        "relative_path": relative,
        "sha256": sha256_file(video),
        "size_bytes": video.stat().st_size,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps > 0 else math.inf,
        "sampled_frames": sampled_frames,
        "motion": {
            "pair_count": len(differences),
            "moving_pair_count": sum(
                value >= MIN_MOTION_MEAN_ABS_DIFFERENCE
                for value in differences
            ),
            "max_mean_abs_difference": max(differences, default=0.0),
        },
    }
    validate_video_metadata(metadata)
    return metadata


def _camera_metadata() -> dict[str, Any]:
    return {
        "cam_prim_path": CAMERA_PRIM_PATH,
        "origin_type": CAMERA_ORIGIN_TYPE,
        "env_index": CAMERA_ENV_INDEX,
        "asset_name": CAMERA_ASSET_NAME,
        "eye": list(CAMERA_EYE),
        "lookat": list(CAMERA_LOOKAT),
        "resolution": [WIDTH, HEIGHT],
    }


def _video_prefix(request: C3VideoRequest) -> str:
    return f"elf3-{request.scenario}-seed{request.seed}-{request.checkpoint.stem}"


def _manifest_payload(
    request: C3VideoRequest,
    binding: Mapping[str, Any],
    *,
    git: Mapping[str, Any],
    configs: Mapping[str, Any],
    assets: Mapping[str, Any],
    m4_sources: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "c3_video",
        "created_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "task_id": TASK_ID,
        "scenario": SCENARIO,
        "seed": SEED,
        "device": DEVICE,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        "cli": request.to_dict(),
        "checkpoint": dict(binding["checkpoint"]),
        "source_evidence": {
            name: dict(binding[name]) for name in ("manifest", "result", "plan")
        },
        "training": dict(binding["training"]),
        "camera": _camera_metadata(),
        "video_contract": {
            "folder": "videos",
            "name_prefix": _video_prefix(request),
            "frame_count": STEPS,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "duration_seconds": DURATION_SECONDS,
            "render_mode": "rgb_array",
            "enable_cameras": True,
        },
        "git": dict(git),
        "configs": dict(configs),
        "assets": dict(assets),
        "m4_sources": dict(m4_sources),
        "runtime": dict(runtime),
        "gpu": dict(gpu),
    }


def _warm_up_renderer(raw_env: Any) -> None:
    for _ in range(RENDER_WARMUP_ATTEMPTS):
        frame = raw_env.render()
        if (
            hasattr(frame, "size")
            and frame.size > 0
            and float(frame.mean()) >= MIN_MEAN_LUMA
        ):
            return
    raise RuntimeError("RGB renderer stayed black during warmup")


def _create_recorded_runner(request, env_cfg, agent_cfg, video_dir: Path):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from openhomie_isaaclab.him_rl.runner import HIMOnPolicyRunner
    from openhomie_isaaclab.workflows import elf3_sim

    env_cfg.viewer.cam_prim_path = CAMERA_PRIM_PATH
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = CAMERA_ENV_INDEX
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = CAMERA_EYE
    env_cfg.viewer.lookat = CAMERA_LOOKAT
    env_cfg.viewer.resolution = (WIDTH, HEIGHT)

    raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    recorded_env = None
    env = None
    try:
        scenario = elf3_sim.EVALUATION_SCENARIOS[SCENARIO]
        raw_env.unwrapped.set_evaluation_command(scenario.command, scenario.mode)
        raw_env.reset()
        _warm_up_renderer(raw_env)
        recorded_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=STEPS,
            name_prefix=_video_prefix(request),
            fps=FPS,
            disable_logger=True,
        )
        env = RslRlVecEnvWrapper(
            recorded_env, clip_actions=agent_cfg.clip_actions
        )
        env = elf3_sim._adapt_bool_dones(env)
        runner = HIMOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=request.device,
        )
    except Exception:
        target = env if env is not None else recorded_env
        if target is None:
            target = raw_env
        target.close()
        raise
    return env, runner


def _load_v3_configs(request: C3VideoRequest, elf3_sim: Any):
    """Resolve the video environment with the final C3 velocity stage."""
    env_cfg, agent_cfg, _ = elf3_sim._load_configs(request)
    env_cfg.command_stage = C3_FINAL_STAGE
    env_mapping = elf3_sim._normalize_json(env_cfg.to_dict())
    agent_mapping = elf3_sim._normalize_json(agent_cfg.to_dict())
    if env_mapping.get("command_stage") != C3_FINAL_STAGE:
        raise RuntimeError("C3 video environment did not resolve the V3 stage")
    configs = {
        "env": env_mapping,
        "agent": agent_mapping,
        "sha256": {
            "env": elf3_sim._json_sha256(env_mapping),
            "agent": elf3_sim._json_sha256(agent_mapping),
        },
    }
    return env_cfg, agent_cfg, configs


def _play_result(request, env, runner, checkpoint_sha256: str) -> dict[str, Any]:
    from openhomie_isaaclab.workflows import elf3_sim

    result = elf3_sim._play_result(
        request,
        env,
        runner,
        checkpoint_sha256,
    )
    if result.get("status") != "PASS" or not isinstance(
        result.get("play"), Mapping
    ):
        raise RuntimeError("forward play did not produce PASS evidence")
    return result


def _append_video_encoder_tail_frame(request, env, runner) -> None:
    """Append one unscored frame so MoviePy emits the fixed 1,000-frame clip."""
    import torch

    from openhomie_isaaclab.workflows import elf3_sim

    policy = runner.get_inference_policy(device=request.device)
    observations = env.get_observations().to(request.device)
    with torch.inference_mode():
        actions = policy(observations)
    expected_shape = (request.num_envs, elf3_sim.C.NUM_POLICY_ACTIONS)
    if tuple(actions.shape) != expected_shape:
        raise RuntimeError("video encoder tail action shape is invalid")
    elf3_sim._require_finite("video encoder tail actions", actions)
    env.step(actions.to(env.device))


def _validate_play_result(result: Mapping[str, Any]) -> None:
    play = result.get("play")
    if not isinstance(play, Mapping):
        raise TypeError("play evidence must be a mapping")
    required = {
        "action_sha256",
        "trajectory_sha256",
        "sha256_before",
        "sha256_after",
    }
    if not required.issubset(play):
        raise ValueError("play hash evidence is incomplete")
    for name in required:
        _valid_sha256(play[name], f"play {name}")


def _record_cleanup_error(current: Exception | None, cleanup: Exception) -> Exception:
    if current is None:
        return cleanup
    return RuntimeError(f"{current}; cleanup also failed: {cleanup}")


def _discover_video(video_dir: Path, request: C3VideoRequest) -> Path:
    expected = video_dir / f"{_video_prefix(request)}-step-0.mp4"
    entries = list(video_dir.iterdir())
    if len(entries) != 1 or entries[0] != expected:
        raise ValueError("video directory must contain exactly the expected MP4")
    return _regular_file(expected, "video MP4")


def run(request: C3VideoRequest) -> dict[str, Any]:
    """Record and verify the fixed C3 clip after Isaac Sim has launched."""
    started = time.perf_counter()
    binding = build_source_binding(request)
    verify_source_binding(binding)

    from openhomie_isaaclab.workflows import elf3_sim

    if elf3_sim.TASK_ID != TASK_ID:
        raise RuntimeError("ELF3 task identity mismatch")
    repo_root = elf3_sim._repository_root()
    git = elf3_sim._git_identity(repo_root)
    elf3_sim._seed_runtime(SEED)
    env_cfg, agent_cfg, configs = _load_v3_configs(request, elf3_sim)
    assets, m4_sources = elf3_sim._source_identity(repo_root)
    runtime = elf3_sim._runtime_identity()
    gpu = elf3_sim._gpu_identity(DEVICE)
    policy_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if not math.isclose(policy_dt, 1.0 / FPS, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("ELF3 policy step does not match the video FPS")
    checkpoint_iteration, checkpoint_payload = elf3_sim._load_checkpoint(
        request.checkpoint, require_optimizers=False
    )
    del checkpoint_payload
    if checkpoint_iteration != binding["checkpoint"]["iteration"]:
        raise RuntimeError("checkpoint payload and source evidence disagree")

    request.output_root.mkdir(mode=0o755)
    video_dir = request.output_root / "videos"
    video_dir.mkdir(mode=0o755)
    manifest = _manifest_payload(
        request,
        binding,
        git=git,
        configs=configs,
        assets=assets,
        m4_sources=m4_sources,
        runtime=runtime,
        gpu=gpu,
    )
    write_json_once(request.output_root / "manifest.json", manifest)

    env = None
    runner = None
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    try:
        env, runner = _create_recorded_runner(
            request, env_cfg, agent_cfg, video_dir
        )
        result = _play_result(
            request,
            env,
            runner,
            binding["checkpoint"]["sha256"],
        )
        _append_video_encoder_tail_frame(request, env, runner)
    except Exception as exc:
        failure = exc
    finally:
        if runner is not None and runner.writer is not None:
            try:
                runner.writer.flush()
                runner.writer.close()
            except Exception as exc:
                failure = _record_cleanup_error(failure, exc)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failure = _record_cleanup_error(failure, exc)

    try:
        verify_source_binding(binding)
        checkpoint_hash_after = sha256_file(request.checkpoint)
        if checkpoint_hash_after != binding["checkpoint"]["sha256"]:
            raise RuntimeError("source checkpoint was mutated")
        if result is not None:
            result["play"]["sha256_after"] = checkpoint_hash_after
    except Exception as exc:
        failure = _record_cleanup_error(failure, exc)

    if failure is None:
        try:
            if result is None:
                raise RuntimeError("video workflow produced no play result")
            video_path = _discover_video(video_dir, request)
            video = inspect_video(video_path, request.output_root)
            _validate_play_result(result)
            result["video"] = video
            result["source_evidence"] = {
                name: dict(binding[name])
                for name in ("checkpoint", "manifest", "result", "plan")
            }
        except Exception as exc:
            failure = exc

    if failure is not None:
        result = {
            "status": "FAIL",
            "failure_code": "C3_VIDEO_RUNTIME_FAILURE",
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    if result is None:
        raise RuntimeError("video workflow produced no result")
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_once(request.output_root / "result.json", result)
    return result
