"""Record a scripted height-and-motion demonstration for the final ELF3 V3 policy."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openhomie_isaaclab.workflows import elf3_c3_video as fixed_video
from openhomie_isaaclab.workflows.elf3_run import sha256_file, write_json_once


SEED = 42
NUM_ENVS = 16
FPS = 50
WIDTH = 1280
HEIGHT = 720
DEVICE = "cuda:0"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
C3_FINAL_STAGE = "V3"

STAND_HEIGHT = 1.01
LOW_HEIGHT = 0.30
TRAINED_MIN_HEIGHT = 0.78
OUT_OF_DISTRIBUTION_HEIGHT = LOW_HEIGHT < TRAINED_MIN_HEIGHT

WALK = 0
HIGH_STAND = 1
CROUCH_LOW = 2

INITIAL_STAND_STEPS = 2 * FPS
CROUCH_TRANSITION_STEPS = 5 * FPS
CROUCH_HOLD_STEPS = 2 * FPS
STAND_TRANSITION_STEPS = 5 * FPS
FORWARD_STEPS = 5 * FPS
MIDDLE_STAND_STEPS = 2 * FPS
BACKWARD_STEPS = 5 * FPS
PRE_TURN_STAND_STEPS = 2 * FPS
TURN_STEPS = 5 * FPS
STEPS = (
    INITIAL_STAND_STEPS
    + CROUCH_TRANSITION_STEPS
    + CROUCH_HOLD_STEPS
    + STAND_TRANSITION_STEPS
    + FORWARD_STEPS
    + MIDDLE_STAND_STEPS
    + BACKWARD_STEPS
    + PRE_TURN_STAND_STEPS
    + TURN_STEPS
)
DURATION_SECONDS = STEPS / FPS

CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
CAMERA_ORIGIN_TYPE = "asset_root"
CAMERA_ENV_INDEX = 0
CAMERA_ASSET_NAME = "robot"
CAMERA_EYE = (3.2, 2.8, 1.8)
CAMERA_LOOKAT = (0.0, 0.0, 0.55)

SAMPLE_FRAME_INDICES = (
    0,
    INITIAL_STAND_STEPS - 1,
    INITIAL_STAND_STEPS,
    INITIAL_STAND_STEPS + CROUCH_TRANSITION_STEPS - 1,
    INITIAL_STAND_STEPS + CROUCH_TRANSITION_STEPS,
    INITIAL_STAND_STEPS + CROUCH_TRANSITION_STEPS + CROUCH_HOLD_STEPS - 1,
    INITIAL_STAND_STEPS + CROUCH_TRANSITION_STEPS + CROUCH_HOLD_STEPS,
    INITIAL_STAND_STEPS
    + CROUCH_TRANSITION_STEPS
    + CROUCH_HOLD_STEPS
    + STAND_TRANSITION_STEPS
    - 1,
    STEPS - 1,
)
MIN_MEAN_LUMA = 1.0
MIN_NONBLACK_FRACTION = 0.01
NONBLACK_LUMA_THRESHOLD = 8
MIN_MOTION_MEAN_ABS_DIFFERENCE = 1.0


@dataclass(frozen=True)
class MotionDemoRequest:
    checkpoint: Path
    output_root: Path
    plan: Path
    plan_sha256: str
    source_manifest: Path
    source_result: Path
    command: str = "c3_motion_demo"
    device: str = DEVICE
    headless: bool = True
    enable_cameras: bool = True
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
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


@dataclass(frozen=True)
class MotionCommand:
    name: str
    command: tuple[float, float, float, float]
    mode: int


def parse_request(argv: Sequence[str] | None = None) -> MotionDemoRequest:
    """Parse the fixed demo request with the same V3 source binding as video."""
    base_request = fixed_video.parse_request(argv)
    return MotionDemoRequest(
        checkpoint=base_request.checkpoint,
        output_root=base_request.output_root,
        plan=base_request.plan,
        plan_sha256=base_request.plan_sha256,
        source_manifest=base_request.source_manifest,
        source_result=base_request.source_result,
    )


def _interpolate(start: float, end: float, index: int, count: int) -> float:
    if not 0 <= index < count:
        raise ValueError("interpolation index is outside the segment")
    if count < 2:
        raise ValueError("interpolation segment must have at least two steps")
    return start + (end - start) * index / (count - 1)


def command_at_step(step: int) -> MotionCommand:
    """Return the fixed user-requested command at one 50 Hz policy step."""
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < STEPS:
        raise ValueError("step is outside the motion demonstration")

    cursor = INITIAL_STAND_STEPS
    if step < cursor:
        return MotionCommand("stand_initial", (0.0, 0.0, 0.0, STAND_HEIGHT), HIGH_STAND)

    start = cursor
    cursor += CROUCH_TRANSITION_STEPS
    if step < cursor:
        height = _interpolate(STAND_HEIGHT, LOW_HEIGHT, step - start, CROUCH_TRANSITION_STEPS)
        return MotionCommand("crouch_down", (0.0, 0.0, 0.0, height), CROUCH_LOW)

    cursor += CROUCH_HOLD_STEPS
    if step < cursor:
        return MotionCommand("crouch_hold", (0.0, 0.0, 0.0, LOW_HEIGHT), CROUCH_LOW)

    start = cursor
    cursor += STAND_TRANSITION_STEPS
    if step < cursor:
        height = _interpolate(LOW_HEIGHT, STAND_HEIGHT, step - start, STAND_TRANSITION_STEPS)
        return MotionCommand("stand_up", (0.0, 0.0, 0.0, height), CROUCH_LOW)

    cursor += FORWARD_STEPS
    if step < cursor:
        return MotionCommand("walk_forward", (0.5, 0.0, 0.0, STAND_HEIGHT), WALK)

    cursor += MIDDLE_STAND_STEPS
    if step < cursor:
        return MotionCommand("stand_middle", (0.0, 0.0, 0.0, STAND_HEIGHT), HIGH_STAND)

    cursor += BACKWARD_STEPS
    if step < cursor:
        return MotionCommand("walk_backward", (-0.3, 0.0, 0.0, STAND_HEIGHT), WALK)

    cursor += PRE_TURN_STAND_STEPS
    if step < cursor:
        return MotionCommand("stand_before_turn", (0.0, 0.0, 0.0, STAND_HEIGHT), HIGH_STAND)

    return MotionCommand("turn_positive", (0.0, 0.0, 0.5, STAND_HEIGHT), WALK)


def timeline() -> list[dict[str, Any]]:
    """Describe the fixed command schedule in a JSON-friendly form."""
    spans: list[tuple[str, int, int]] = []
    start = 0
    for name, count in (
        ("stand_initial", INITIAL_STAND_STEPS),
        ("crouch_down", CROUCH_TRANSITION_STEPS),
        ("crouch_hold", CROUCH_HOLD_STEPS),
        ("stand_up", STAND_TRANSITION_STEPS),
        ("walk_forward", FORWARD_STEPS),
        ("stand_middle", MIDDLE_STAND_STEPS),
        ("walk_backward", BACKWARD_STEPS),
        ("stand_before_turn", PRE_TURN_STAND_STEPS),
        ("turn_positive", TURN_STEPS),
    ):
        spans.append((name, start, start + count))
        start += count
    if start != STEPS:
        raise RuntimeError("motion demonstration duration is inconsistent")
    return [
        {
            "name": name,
            "start_step": begin,
            "end_step_exclusive": end,
            "duration_seconds": (end - begin) / FPS,
            "mode": command_at_step(begin).mode,
            "command_start": list(command_at_step(begin).command),
            "command_end": list(command_at_step(end - 1).command),
        }
        for name, begin, end in spans
    ]


def _video_prefix(request: MotionDemoRequest) -> str:
    return f"elf3-motion-demo-seed{request.seed}-{request.checkpoint.stem}"


def _create_recorded_runner(request, env_cfg, agent_cfg, video_dir: Path):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from openhomie_isaaclab.him_rl.runner import HIMOnPolicyRunner
    from openhomie_isaaclab.workflows import elf3_sim

    env_cfg.viewer.cam_prim_path = CAMERA_PRIM_PATH
    env_cfg.viewer.origin_type = CAMERA_ORIGIN_TYPE
    env_cfg.viewer.env_index = CAMERA_ENV_INDEX
    env_cfg.viewer.asset_name = CAMERA_ASSET_NAME
    env_cfg.viewer.eye = CAMERA_EYE
    env_cfg.viewer.lookat = CAMERA_LOOKAT
    env_cfg.viewer.resolution = (WIDTH, HEIGHT)

    raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    recorded_env = None
    env = None
    try:
        initial = command_at_step(0)
        raw_env.unwrapped.set_evaluation_command(initial.command, initial.mode)
        raw_env.reset()
        fixed_video._warm_up_renderer(raw_env)
        recorded_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=STEPS,
            name_prefix=_video_prefix(request),
            fps=FPS,
            disable_logger=True,
        )
        env = elf3_sim._adapt_bool_dones(
            RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
        )
        runner = HIMOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=request.device
        )
    except Exception:
        target = env if env is not None else recorded_env
        if target is None:
            target = raw_env
        target.close()
        raise
    return env, runner


def _record_observation(
    stats: dict[str, dict[str, float | int]], name: str, snapshot: Mapping[str, Any]
) -> None:
    height = float(snapshot["tracking_height"][0].item())
    velocity = float(snapshot["root_lin_vel_b"][0, 0].item())
    yaw_rate = float(snapshot["root_ang_vel_b"][0, 2].item())
    current = stats.setdefault(
        name,
        {
            "samples": 0,
            "tracking_height_min": math.inf,
            "tracking_height_max": -math.inf,
            "tracking_height_sum": 0.0,
            "linear_x_velocity_sum": 0.0,
            "yaw_rate_sum": 0.0,
        },
    )
    current["samples"] += 1
    current["tracking_height_min"] = min(current["tracking_height_min"], height)
    current["tracking_height_max"] = max(current["tracking_height_max"], height)
    current["tracking_height_sum"] += height
    current["linear_x_velocity_sum"] += velocity
    current["yaw_rate_sum"] += yaw_rate


def _summarize_observations(stats: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, current in stats.items():
        samples = int(current["samples"])
        if samples <= 0:
            raise RuntimeError("motion phase has no observed samples")
        result[name] = {
            "samples": samples,
            "tracking_height_min": float(current["tracking_height_min"]),
            "tracking_height_max": float(current["tracking_height_max"]),
            "tracking_height_mean": float(current["tracking_height_sum"]) / samples,
            "linear_x_velocity_mean": float(current["linear_x_velocity_sum"]) / samples,
            "yaw_rate_mean": float(current["yaw_rate_sum"]) / samples,
        }
    return result


def _run_timeline(request, env, runner) -> dict[str, Any]:
    import torch

    from openhomie_isaaclab.workflows import elf3_sim

    runner.load(str(request.checkpoint), load_optimizer=False, map_location=request.device)
    runner.eval_mode()
    policy = runner.get_inference_policy(device=request.device)
    observations = env.get_observations().to(request.device)
    raw_env = env.unwrapped
    stats: dict[str, dict[str, float | int]] = {}
    expected_shape = (NUM_ENVS, elf3_sim.C.NUM_POLICY_ACTIONS)
    with torch.inference_mode():
        for step in range(STEPS):
            entry = command_at_step(step)
            raw_env.set_evaluation_command(entry.command, entry.mode)
            snapshot = raw_env.get_evaluation_observables()
            _record_observation(stats, entry.name, snapshot)
            elf3_sim._require_finite("motion demonstration observations", observations)
            actions = policy(observations)
            if tuple(actions.shape) != expected_shape:
                raise RuntimeError("motion demonstration action shape is invalid")
            elf3_sim._require_finite("motion demonstration actions", actions)
            observations, rewards, _, _ = env.step(actions.to(env.device))
            observations = observations.to(request.device)
            elf3_sim._require_finite("motion demonstration rewards", rewards)

        # Gymnasium/MoviePy emits one fewer encoded frame at this FPS unless the
        # wrapper receives one unscored tail action.
        tail = command_at_step(STEPS - 1)
        raw_env.set_evaluation_command(tail.command, tail.mode)
        actions = policy(observations)
        if tuple(actions.shape) != expected_shape:
            raise RuntimeError("motion demonstration tail action shape is invalid")
        elf3_sim._require_finite("motion demonstration tail actions", actions)
        env.step(actions.to(env.device))
    return {
        "steps": STEPS,
        "observed_env0": _summarize_observations(stats),
    }


def _inspect_video(video_path: Path, output_root: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    video = fixed_video._regular_file(video_path, "motion demonstration MP4")
    root = output_root.resolve(strict=True)
    try:
        relative_path = video.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("motion demonstration MP4 escaped its output root") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("motion demonstration MP4 cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    sampled: dict[int, Any] = {}
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tuple(frame.shape[:2]) != (HEIGHT, WIDTH):
                raise ValueError("motion demonstration frame dimensions changed")
            if frame_count in SAMPLE_FRAME_INDICES:
                sampled[frame_count] = frame.copy()
            frame_count += 1
    finally:
        capture.release()
    if frame_count != STEPS:
        raise ValueError("motion demonstration MP4 frame count is wrong")
    if set(sampled) != set(SAMPLE_FRAME_INDICES):
        raise ValueError("motion demonstration MP4 is missing sampled frames")
    if not math.isclose(fps, FPS, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("motion demonstration MP4 FPS is wrong")
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError("motion demonstration MP4 dimensions are wrong")

    frames = [sampled[index] for index in SAMPLE_FRAME_INDICES]
    samples = []
    for index, frame in zip(SAMPLE_FRAME_INDICES, frames, strict=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        nonblack_fraction = float(
            np.count_nonzero(gray > NONBLACK_LUMA_THRESHOLD) / gray.size
        )
        if mean_luma < MIN_MEAN_LUMA or nonblack_fraction < MIN_NONBLACK_FRACTION:
            raise ValueError("motion demonstration MP4 contains a black sampled frame")
        samples.append(
            {
                "index": index,
                "mean_luma": mean_luma,
                "nonblack_fraction": nonblack_fraction,
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
        for previous, current in zip(frames, frames[1:])
    ]
    if max(differences, default=0.0) < MIN_MOTION_MEAN_ABS_DIFFERENCE:
        raise ValueError("motion demonstration MP4 lacks visible motion")
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(video),
        "size_bytes": video.stat().st_size,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
        "sampled_frames": samples,
        "motion": {
            "pair_count": len(differences),
            "moving_pair_count": sum(
                value >= MIN_MOTION_MEAN_ABS_DIFFERENCE for value in differences
            ),
            "max_mean_abs_difference": max(differences, default=0.0),
        },
    }


def _discover_video(video_dir: Path, request: MotionDemoRequest) -> Path:
    expected = video_dir / f"{_video_prefix(request)}-step-0.mp4"
    entries = list(video_dir.iterdir())
    if len(entries) != 1 or entries[0] != expected:
        raise ValueError("motion demonstration video directory is not exact")
    return expected


def _manifest(request, binding, configs, timeline_spec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": request.command,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": dict(binding["checkpoint"]),
        "source_evidence": {
            name: dict(binding[name]) for name in ("manifest", "result", "plan")
        },
        "training": dict(binding["training"]),
        "cli": request.to_dict(),
        "camera": {
            "cam_prim_path": CAMERA_PRIM_PATH,
            "origin_type": CAMERA_ORIGIN_TYPE,
            "env_index": CAMERA_ENV_INDEX,
            "asset_name": CAMERA_ASSET_NAME,
            "eye": list(CAMERA_EYE),
            "lookat": list(CAMERA_LOOKAT),
            "resolution": [WIDTH, HEIGHT],
        },
        "timeline": timeline_spec,
        "out_of_distribution": {
            "requested_low_height": LOW_HEIGHT,
            "trained_min_height": TRAINED_MIN_HEIGHT,
            "is_out_of_distribution": OUT_OF_DISTRIBUTION_HEIGHT,
        },
        "configs": configs,
        "video_contract": {
            "frame_count": STEPS,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "duration_seconds": DURATION_SECONDS,
        },
    }


def run(request: MotionDemoRequest) -> dict[str, Any]:
    """Record the scripted demonstration after Isaac Sim has launched."""
    started = time.perf_counter()
    binding = fixed_video.build_source_binding(request)
    fixed_video.verify_source_binding(binding)

    from openhomie_isaaclab.workflows import elf3_sim

    elf3_sim._seed_runtime(request.seed)
    env_cfg, agent_cfg, configs = fixed_video._load_v3_configs(request, elf3_sim)
    if not math.isclose(
        float(env_cfg.sim.dt * env_cfg.decimation),
        1.0 / FPS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("motion demonstration policy step does not match FPS")
    checkpoint_iteration, payload = elf3_sim._load_checkpoint(
        request.checkpoint, require_optimizers=False
    )
    del payload
    if checkpoint_iteration != binding["checkpoint"]["iteration"]:
        raise RuntimeError("motion demonstration checkpoint identity disagrees")

    request.output_root.mkdir(mode=0o755)
    video_dir = request.output_root / "videos"
    video_dir.mkdir(mode=0o755)
    timeline_spec = timeline()
    write_json_once(
        request.output_root / "manifest.json",
        _manifest(request, binding, configs, timeline_spec),
    )

    env = None
    runner = None
    failure: Exception | None = None
    play: dict[str, Any] | None = None
    try:
        env, runner = _create_recorded_runner(request, env_cfg, agent_cfg, video_dir)
        play = _run_timeline(request, env, runner)
    except Exception as exc:
        failure = exc
    finally:
        if runner is not None and runner.writer is not None:
            try:
                runner.writer.flush()
                runner.writer.close()
            except Exception as exc:
                failure = fixed_video._record_cleanup_error(failure, exc)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failure = fixed_video._record_cleanup_error(failure, exc)

    try:
        fixed_video.verify_source_binding(binding)
        if sha256_file(request.checkpoint) != binding["checkpoint"]["sha256"]:
            raise RuntimeError("motion demonstration source checkpoint was mutated")
    except Exception as exc:
        failure = fixed_video._record_cleanup_error(failure, exc)

    if failure is None:
        try:
            if play is None:
                raise RuntimeError("motion demonstration produced no play record")
            video = _inspect_video(_discover_video(video_dir, request), request.output_root)
            result: dict[str, Any] = {
                "status": "PASS",
                "source_evidence": {
                    name: dict(binding[name])
                    for name in ("checkpoint", "manifest", "result", "plan")
                },
                "timeline": timeline_spec,
                "out_of_distribution": {
                    "requested_low_height": LOW_HEIGHT,
                    "trained_min_height": TRAINED_MIN_HEIGHT,
                    "is_out_of_distribution": OUT_OF_DISTRIBUTION_HEIGHT,
                },
                "play": play,
                "video": video,
            }
        except Exception as exc:
            failure = exc
    if failure is not None:
        result = {
            "status": "FAIL",
            "failure_code": "C3_MOTION_DEMO_RUNTIME_FAILURE",
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_once(request.output_root / "result.json", result)
    return result
