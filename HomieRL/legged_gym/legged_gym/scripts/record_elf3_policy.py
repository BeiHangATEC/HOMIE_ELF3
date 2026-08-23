#!/usr/bin/env python3
"""Record a deterministic ELF3 policy demonstration in Isaac Gym."""

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from isaacgym import gymapi, gymutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils import get_args, task_registry


FPS = 25
WIDTH = 1280
HEIGHT = 720
STAND_HEIGHT = 1.01
SQUAT_HEIGHT = 0.30
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

# End time, label, vx, vy, yaw rate, base height, transition duration.
TIMELINE = [
    (3.0, "站立准备", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.0),
    (7.0, "下蹲", 0.0, 0.0, 0.0, SQUAT_HEIGHT, 4.0),
    (9.0, "保持下蹲", 0.0, 0.0, 0.0, SQUAT_HEIGHT, 0.0),
    (13.0, "起立", 0.0, 0.0, 0.0, STAND_HEIGHT, 4.0),
    (15.0, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (19.5, "前进", 0.70, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (21.0, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (25.5, "后退", -0.60, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (27.0, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (31.0, "原地左转", 0.0, 0.0, 0.45, STAND_HEIGHT, 0.8),
    (32.5, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (36.5, "原地右转", 0.0, 0.0, -0.45, STAND_HEIGHT, 0.8),
    (38.0, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (42.5, "前进左转弯", 0.55, 0.0, 0.35, STAND_HEIGHT, 0.8),
    (44.0, "站稳", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
    (48.5, "前进右转弯", 0.55, 0.0, -0.35, STAND_HEIGHT, 0.8),
    (51.5, "站立结束", 0.0, 0.0, 0.0, STAND_HEIGHT, 0.8),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_recording_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=TIMELINE[-1][0])
    parser.add_argument("--video-fps", type=int, default=FPS)
    parser.add_argument("--video-width", type=int, default=WIDTH)
    parser.add_argument("--video-height", type=int, default=HEIGHT)
    record_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return record_args


def patch_headless_rendering():
    """Keep a graphics device for camera sensors while omitting the GUI viewer."""

    def patched_init(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        self.graphics_device_id = self.sim_device_id
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_dofs
        self.num_one_step_obs = cfg.env.num_one_step_observations
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
        self.extras = {}

        self.create_sim()
        if self.sim is None:
            raise RuntimeError("Isaac Gym failed to create the simulation")
        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = False
        self.viewer = None

    BaseTask.__init__ = patched_init


def configure_environment(env_cfg, duration):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(duration + 5.0, 60.0)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = max(duration + 5.0, 60.0)
    if hasattr(env_cfg.commands, "use_random"):
        env_cfg.commands.use_random = False
    if hasattr(env_cfg.asset, "self_collision"):
        env_cfg.asset.self_collision = 0
    if hasattr(env_cfg.asset, "self_collisions"):
        env_cfg.asset.self_collisions = 0
    env_cfg.asset.terminate_after_contacts_on = []
    if hasattr(env_cfg.env, "upper_teleop"):
        env_cfg.env.upper_teleop = False

    domain_rand = env_cfg.domain_rand
    for name in dir(domain_rand):
        if name.startswith("randomize_"):
            value = getattr(domain_rand, name)
            if isinstance(value, bool):
                setattr(domain_rand, name, False)
    for name in ("use_random", "push_robots", "disturbance", "delay"):
        if hasattr(domain_rand, name):
            setattr(domain_rand, name, False)


def unique_output_path(path):
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a free output name beside {path}")


def command_at(sim_time):
    previous_end = 0.0
    previous = np.array([0.0, 0.0, 0.0, STAND_HEIGHT], dtype=np.float32)
    for end, label, vx, vy, yaw, height, transition_duration in TIMELINE:
        target = np.array([vx, vy, yaw, height], dtype=np.float32)
        if sim_time < end:
            elapsed = sim_time - previous_end
            transition = min(transition_duration, max(0.0, end - previous_end))
            if transition > 0.0 and elapsed < transition:
                ratio = 0.5 - 0.5 * math.cos(math.pi * elapsed / transition)
                command = previous + (target - previous) * ratio
            else:
                command = target
            return label, command
        previous_end = end
        previous = target
    return TIMELINE[-1][1], previous


def yaw_from_xyzw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@functools.lru_cache(maxsize=None)
def latin_font(size):
    if LATIN_FONT_PATH.exists():
        return ImageFont.truetype(str(LATIN_FONT_PATH), size=size)
    return ImageFont.load_default()


@functools.lru_cache(maxsize=None)
def cjk_font(size):
    if CJK_FONT_PATH.exists():
        return ImageFont.truetype(str(CJK_FONT_PATH), size=size)
    return ImageFont.load_default()


def draw_overlay(frame, sim_time, duration, label, command, checkpoint_iteration):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rounded_rectangle((24, 20, 620, 112), radius=14, fill=(5, 12, 20, 188))
    draw.text((44, 30), "HOMIE ELF3", font=latin_font(30), fill=(255, 255, 255, 255))
    draw.text((248, 30), "策略演示", font=cjk_font(30), fill=(255, 255, 255, 255))
    draw.text(
        (44, 72),
        f"checkpoint: model_{checkpoint_iteration}.pt",
        font=latin_font(20),
        fill=(176, 213, 255, 255),
    )
    draw.rounded_rectangle(
        (24, height - 128, width - 24, height - 22),
        radius=14,
        fill=(5, 12, 20, 195),
    )
    draw.text((46, height - 116), label, font=cjk_font(36), fill=(255, 221, 92, 255))
    draw.text(
        (280, height - 108),
        f"vx {command[0]:+.2f} m/s   vy {command[1]:+.2f} m/s   "
        f"yaw {command[2]:+.2f} rad/s   height {command[3]:.2f} m",
        font=latin_font(22),
        fill=(235, 241, 248, 255),
    )
    progress_x0 = 46
    progress_x1 = width - 46
    progress_y = height - 43
    draw.rounded_rectangle(
        (progress_x0, progress_y, progress_x1, progress_y + 8),
        radius=4,
        fill=(84, 94, 108, 230),
    )
    ratio = min(max(sim_time / duration, 0.0), 1.0)
    draw.rounded_rectangle(
        (
            progress_x0,
            progress_y,
            progress_x0 + (progress_x1 - progress_x0) * ratio,
            progress_y + 8,
        ),
        radius=4,
        fill=(73, 181, 255, 255),
    )
    draw.text(
        (width - 150, 28),
        f"{sim_time:04.1f}s",
        font=latin_font(25),
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 210),
    )
    return np.asarray(image)


def open_encoder(output, width, height, fps):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def segment_definition():
    start = 0.0
    definitions = []
    for end, label, vx, vy, yaw, height, transition_duration in TIMELINE:
        definitions.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "target_command": {
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "yaw_rate_radps": yaw,
                    "height_m": height,
                },
                "transition_duration_seconds": transition_duration,
            }
        )
        start = end
    return definitions


def main():
    record_args = parse_recording_args()
    if record_args.duration <= 0.0 or record_args.duration > TIMELINE[-1][0]:
        raise ValueError(f"--duration must be in (0, {TIMELINE[-1][0]}]")
    checkpoint = record_args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    checkpoint_payload = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_iteration = int(checkpoint_payload.get("iter", -1))
    if checkpoint_iteration < 0:
        raise ValueError(f"Checkpoint has no valid iteration: {checkpoint}")
    checkpoint_stat = checkpoint.stat()
    checkpoint_sha256 = sha256_file(checkpoint)
    checkpoint_metadata_path = checkpoint.with_suffix(".json")
    checkpoint_metadata = None
    if checkpoint_metadata_path.is_file():
        checkpoint_metadata = json.loads(
            checkpoint_metadata_path.read_text(encoding="utf-8")
        )

    output = unique_output_path(record_args.output.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    recording_started_at = utc_now()

    args = get_args()
    args.task = "elf3"
    args.num_envs = 1
    args.headless = True
    args.resume = False
    patch_headless_rendering()

    env_cfg, train_cfg = task_registry.get_cfgs(name="elf3")
    configure_environment(env_cfg, record_args.duration)
    env, _ = task_registry.make_env(name="elf3", args=args, env_cfg=env_cfg)
    env.action_curriculum_ratio = 0.0
    env._resample_commands = lambda env_ids: None

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name="elf3",
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    initial_command = command_at(0.0)[1]
    env.commands[:, 0] = float(initial_command[0])
    env.commands[:, 1] = float(initial_command[1])
    env.commands[:, 2] = float(initial_command[2])
    env.commands[:, 4] = float(initial_command[3])
    env.reset_buf[:] = 0
    env.time_out_buf[:] = False
    env.gravity_termination_buf[:] = False
    env.compute_observations()
    obs = env.get_observations()

    camera_props = gymapi.CameraProperties()
    camera_props.width = record_args.video_width
    camera_props.height = record_args.video_height
    camera_props.horizontal_fov = 58.0
    camera_props.enable_tensors = False
    camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    if camera < 0:
        raise RuntimeError("Isaac Gym failed to create the recording camera")

    encoder = open_encoder(
        output,
        record_args.video_width,
        record_args.video_height,
        record_args.video_fps,
    )

    total_steps = int(math.ceil(record_args.duration / env.dt))
    next_frame_time = 0.0
    frame_count = 0
    reset_times = []
    min_root_height = float("inf")
    max_root_height = float("-inf")
    started = time.time()

    try:
        for step in range(total_steps):
            sim_time = step * env.dt
            label, command = command_at(sim_time)

            env.commands[:, 0] = float(command[0])
            env.commands[:, 1] = float(command[1])
            env.commands[:, 2] = float(command[2])
            env.commands[:, 4] = float(command[3])
            env.action_curriculum_ratio = 0.0

            with torch.inference_mode():
                actions = policy(obs.detach())
                obs, _, _, dones, _, _, _ = env.step(actions.detach())

            if int(dones[0].item()) != 0:
                reset_times.append(sim_time)
                raise RuntimeError(f"Environment reset at {sim_time:.2f}s")

            root = env.root_states[0].detach().cpu().numpy()
            min_root_height = min(min_root_height, float(root[2]))
            max_root_height = max(max_root_height, float(root[2]))

            if sim_time + 1e-9 >= next_frame_time:
                yaw = yaw_from_xyzw(root[3:7])
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                body_offset_x = -3.2
                body_offset_y = -2.7
                camera_x = root[0] + cos_yaw * body_offset_x - sin_yaw * body_offset_y
                camera_y = root[1] + sin_yaw * body_offset_x + cos_yaw * body_offset_y
                camera_z = max(1.9, root[2] + 1.15)
                target_z = max(0.65, root[2] * 0.65)
                env.gym.set_camera_location(
                    camera,
                    env.envs[0],
                    gymapi.Vec3(float(camera_x), float(camera_y), float(camera_z)),
                    gymapi.Vec3(float(root[0]), float(root[1]), float(target_z)),
                )
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                rgba = env.gym.get_camera_image(
                    env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
                )
                rgba = np.asarray(rgba).reshape(
                    record_args.video_height, record_args.video_width, 4
                )
                frame = np.ascontiguousarray(rgba[:, :, :3])
                frame = draw_overlay(
                    frame,
                    sim_time,
                    record_args.duration,
                    label,
                    command,
                    checkpoint_iteration,
                )
                try:
                    encoder.stdin.write(frame.tobytes())
                except BrokenPipeError as error:
                    raise RuntimeError("ffmpeg stopped while encoding the video") from error
                frame_count += 1
                next_frame_time = frame_count / record_args.video_fps
    finally:
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()

    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    final_root = env.root_states[0].detach().cpu().numpy()
    metadata = {
        "schema_version": 1,
        "robot": "ELF3",
        "task": "elf3",
        "evaluation": "nominal: plane, fixed upper body, randomization disabled",
        "simulator": "Isaac Gym Preview 4",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "recording_started_at_utc": recording_started_at,
        "recording_completed_at_utc": utc_now(),
        "output": str(output),
        "duration_seconds": record_args.duration,
        "fps": record_args.video_fps,
        "frames": frame_count,
        "resolution": [record_args.video_width, record_args.video_height],
        "environment_resets": len(reset_times),
        "reset_times": reset_times,
        "min_root_height": min_root_height,
        "max_root_height": max_root_height,
        "final_base_position": final_root[:3].tolist(),
        "wall_time_seconds": time.time() - started,
        "segments": segment_definition(),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Video saved to: {output}")
    print(f"Metadata saved to: {metadata_path}")
    if reset_times:
        raise RuntimeError(f"Recording had {len(reset_times)} environment resets")

    env.gym.destroy_sim(env.sim)
    env.sim = None
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
