"""Isaac Lab DirectRLEnv implementation for the 28-DOF ELF3 robot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from openhomie_isaaclab import elf3_constants as C

from . import elf3_homie_curriculum as curriculum
from . import elf3_homie_rewards as R
from . import elf3_stages
from .elf3_homie_env_cfg import Elf3HomieEnvCfg
from .elf3_homie_env_core import (
    apply_control_randomization,
    assemble_actor_frame,
    build_name_permutation,
    classify_dones,
    compute_leg_efforts,
    gather_canonical,
    scatter_canonical,
    shift_history,
    virtual_sole_corners,
)


HIP_JOINT_NAMES = (
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
)
ANKLE_ROLL_JOINT_NAMES = ("l_ankle_x_joint", "r_ankle_x_joint")
KNEE_JOINT_NAMES = ("l_knee_y_joint", "r_knee_y_joint")


def _indices(names: Sequence[str]) -> tuple[int, ...]:
    return tuple(C.joint_index(name) for name in names)


def _gain(name: str, table: Mapping[str, float]) -> float:
    matches = [float(value) for pattern, value in table.items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise RuntimeError(f"ELF3 joint {name!r} must match exactly one gain")
    return matches[0]


class Elf3HomieEnv(DirectRLEnv):
    cfg: Elf3HomieEnvCfg

    def __init__(self, cfg: Elf3HomieEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_layout()
        self._allocate_state()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _resolve_layout(self):
        permutation = build_name_permutation(self._robot.joint_names, C.JOINT_NAMES)
        self._canonical_to_runtime = permutation.to(self.device)
        inverse = torch.empty_like(self._canonical_to_runtime)
        inverse[self._canonical_to_runtime] = torch.arange(C.NUM_ROBOT_DOFS, device=self.device)
        self._runtime_to_canonical = inverse
        self._leg_canonical = torch.as_tensor(C.LOWER_DOF_INDICES, dtype=torch.long, device=self.device)
        self._upper_canonical = torch.as_tensor(C.UPPER_DOF_INDICES, dtype=torch.long, device=self.device)
        self._leg_runtime = self._canonical_to_runtime[self._leg_canonical]
        self._upper_runtime = self._canonical_to_runtime[self._upper_canonical]

        self._imu_id = self._one_body(C.IMU_BODY_NAME)
        self._torso_id = self._one_body(C.TORSO_BODY_NAME)
        self._feet_ids = self._bodies(C.FOOT_BODY_NAMES)
        self._knee_body_ids = self._bodies(C.KNEE_BODY_NAMES)
        contact_ids, contact_names = self._contact_sensor.find_bodies(C.FOOT_BODY_NAMES, preserve_order=True)
        if contact_names != list(C.FOOT_BODY_NAMES):
            raise RuntimeError(f"ELF3 contact foot mapping mismatch: {contact_names}")
        self._feet_contact_ids = torch.as_tensor(contact_ids, dtype=torch.long, device=self.device)
        torso_ids, torso_names = self._contact_sensor.find_bodies(C.TORSO_BODY_NAME, preserve_order=True)
        if torso_names != [C.TORSO_BODY_NAME]:
            raise RuntimeError(f"ELF3 torso contact mapping mismatch: {torso_names}")
        self._torso_contact_ids = torch.as_tensor(torso_ids, dtype=torch.long, device=self.device)

        actuator_names = {
            name for actuator in self._robot.actuators.values() for name in actuator.joint_names
        }
        if actuator_names != set(C.JOINT_NAMES):
            raise RuntimeError("ELF3 actuators do not cover the canonical joint set")
        if set(self._robot.actuators["legs"].joint_names) != set(C.POLICY_JOINT_NAMES):
            raise RuntimeError("ELF3 leg actuator group does not match policy joints")

    def _one_body(self, name: str) -> int:
        ids, names = self._robot.find_bodies(name, preserve_order=True)
        if names != [name]:
            raise RuntimeError(f"ELF3 body mapping mismatch for {name}: {names}")
        return ids[0]

    def _bodies(self, names: Sequence[str]) -> torch.Tensor:
        ids, resolved = self._robot.find_bodies(list(names), preserve_order=True)
        if resolved != list(names):
            raise RuntimeError(f"ELF3 body mapping mismatch: {resolved}")
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _allocate_state(self):
        n = self.num_envs
        self._default_pos = gather_canonical(
            self._robot.data.default_joint_pos, self._canonical_to_runtime
        ).clone()
        hard = self._robot.data.joint_pos_limits[:, self._canonical_to_runtime].clone()
        soft = self._robot.data.soft_joint_pos_limits[:, self._canonical_to_runtime].clone()
        self._hard_lower, self._hard_upper = hard.unbind(-1)
        self._soft_lower, self._soft_upper = soft.unbind(-1)
        self._velocity_limits = gather_canonical(
            self._robot.data.joint_vel_limits, self._canonical_to_runtime
        ).clone()
        self._effort_limits = self._resolve_effort_limits()
        for label, tensor in (("velocity", self._velocity_limits), ("effort", self._effort_limits)):
            if tensor.shape != (n, C.NUM_ROBOT_DOFS) or not torch.all(torch.isfinite(tensor) & (tensor > 0)):
                raise RuntimeError(f"ELF3 {label} limits must be finite, positive, and canonical")

        self._kp = torch.tensor(
            [_gain(name, C.ARMATURE and (C.LEG_STIFFNESS | C.UPPER_BODY_STIFFNESS)) for name in C.JOINT_NAMES],
            device=self.device,
        ).unsqueeze(0)
        self._kd = torch.tensor(
            [_gain(name, C.LEG_DAMPING | C.UPPER_BODY_DAMPING) for name in C.JOINT_NAMES],
            device=self.device,
        ).unsqueeze(0)
        self._action_min = (self._hard_lower[:, self._leg_canonical] - self._default_pos[:, self._leg_canonical]) / C.ACTION_SCALE
        self._action_max = (self._hard_upper[:, self._leg_canonical] - self._default_pos[:, self._leg_canonical]) / C.ACTION_SCALE

        self._actions = torch.zeros(n, C.NUM_POLICY_ACTIONS, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._second_previous_actions = torch.zeros_like(self._actions)
        self._last_runtime_actions = torch.zeros(n, C.NUM_ROBOT_DOFS, device=self.device)
        self._delayed_runtime_actions = torch.zeros(C.DECIMATION, n, C.NUM_ROBOT_DOFS, device=self.device)
        self._delay_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._control_substep = 0
        self._upper_actions = torch.zeros(n, C.NUM_UPPER_BODY_DOFS, device=self.device)
        self._upper_target = torch.zeros_like(self._upper_actions)
        self._upper_delta = torch.zeros_like(self._upper_actions)
        self.action_curriculum_ratio = float(self.cfg.action_curriculum_ratio)

        self._joint_target = self._default_pos.clone()
        self._kp_factors = torch.ones(n, C.NUM_ROBOT_DOFS, device=self.device)
        self._kd_factors = torch.ones_like(self._kp_factors)
        self._actuation_offset = torch.zeros_like(self._kp_factors)
        self._joint_injection = torch.zeros_like(self._kp_factors)
        self._applied_torques = torch.zeros_like(self._kp_factors)
        self._last_dof_vel = torch.zeros_like(self._kp_factors)

        self._commands = torch.zeros(n, 5, device=self.device)
        self._modes = torch.zeros(n, dtype=torch.long, device=self.device)
        self._stage = elf3_stages.get_stage(self.cfg.command_stage)
        self._command_steps = max(1, round(self.cfg.command_resampling_time_s / self.step_dt))
        self._upper_steps = max(1, round(self.cfg.upper_body_resampling_time_s / self.step_dt))
        if self.cfg.upper_body_interpolation_steps != self._upper_steps:
            raise ValueError("upper body interpolation must span one resampling interval")
        if not 0 <= self.cfg.max_action_delay_substeps < C.DECIMATION:
            raise ValueError("action delay must be smaller than decimation")

        self._history = torch.zeros(n, C.NUM_ACTOR_HISTORY, C.num_one_step_actor_obs(), device=self.device)
        self._history_reset_ids = torch.arange(n, device=self.device)
        self._last_history_step = -1
        self._noise = self._build_noise()
        self._gravity_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(n, 1)
        self._last_contacts = torch.zeros(n, 2, dtype=torch.bool, device=self.device)
        self._contact_filter = torch.zeros_like(self._last_contacts)
        self._first_contacts = torch.zeros_like(self._last_contacts)
        self._air_time = torch.zeros(n, 2, device=self.device)
        self._contact_forces = torch.zeros(n, 2, 3, device=self.device)
        self._episode_sums = {name: torch.zeros(n, device=self.device) for name in R.REWARD_NAMES}
        self.last_raw_reward_terms = {name: torch.zeros(n, device=self.device) for name in R.REWARD_NAMES}
        self.last_scaled_reward_terms = {name: torch.zeros(n, device=self.device) for name in R.REWARD_NAMES}

    def _resolve_effort_limits(self) -> torch.Tensor:
        runtime = torch.empty(self.num_envs, C.NUM_ROBOT_DOFS, device=self.device)
        for actuator in self._robot.actuators.values():
            ids = actuator.joint_indices
            if isinstance(ids, slice):
                ids = torch.arange(C.NUM_ROBOT_DOFS, device=self.device)[ids]
            else:
                ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            limits = actuator.effort_limit
            if limits.ndim == 1:
                limits = limits.unsqueeze(0).expand(self.num_envs, -1)
            runtime[:, ids] = limits
        return gather_canonical(runtime, self._canonical_to_runtime)

    def _build_noise(self) -> torch.Tensor:
        noise = torch.zeros(C.num_one_step_actor_obs(), device=self.device)
        cursor = C.NUM_COMMAND_OBS
        noise[cursor : cursor + 3] = self.cfg.ang_vel_noise * self.cfg.noise_level * C.ANG_VEL_SCALE
        cursor += 3
        noise[cursor : cursor + 3] = self.cfg.gravity_noise * self.cfg.noise_level
        cursor = C.NUM_OBS_HEAD
        noise[cursor : cursor + C.NUM_ROBOT_DOFS] = self.cfg.dof_pos_noise * self.cfg.noise_level * C.DOF_POS_SCALE
        cursor += C.NUM_ROBOT_DOFS
        noise[cursor : cursor + C.NUM_ROBOT_DOFS] = self.cfg.dof_vel_noise * self.cfg.noise_level * C.DOF_VEL_SCALE
        return noise

    def _update_upper_body(self):
        if self.cfg.freeze_upper_body:
            self._upper_actions.zero_()
            return
        if self.common_step_counter % self._upper_steps == 0:
            low = (self._hard_lower[:, self._upper_canonical] - self._default_pos[:, self._upper_canonical]) / C.ACTION_SCALE
            high = (self._hard_upper[:, self._upper_canonical] - self._default_pos[:, self._upper_canonical]) / C.ACTION_SCALE
            self._upper_target.copy_(curriculum.sample_upper_body_targets(
                action_min=low, action_max=high,
                curriculum_ratio=self.action_curriculum_ratio,
                amplitude_draws=torch.rand_like(low), joint_draws=torch.rand_like(low),
                direction_draws=torch.rand_like(low),
            ))
            self._upper_delta.copy_(curriculum.upper_body_interpolation_delta(
                self._upper_target, self._upper_actions, self.cfg.upper_body_interpolation_steps
            ))
        self._upper_actions.add_(self._upper_delta)

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, C.NUM_POLICY_ACTIONS):
            raise ValueError(f"ELF3 actions must have shape {(self.num_envs, C.NUM_POLICY_ACTIONS)}")
        self._assert_finite(actions, "actions")
        self._update_upper_body()
        self._actions.copy_(torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions))
        canonical = torch.zeros(self.num_envs, C.NUM_ROBOT_DOFS, device=self.device)
        canonical[:, self._leg_canonical] = self._actions
        canonical[:, self._upper_canonical] = self._upper_actions
        runtime = scatter_canonical(canonical, self._canonical_to_runtime, C.NUM_ROBOT_DOFS)
        if self.cfg.enable_action_delay:
            self._delay_steps.random_(0, self.cfg.max_action_delay_substeps + 1)
        else:
            self._delay_steps.zero_()
        substeps = torch.arange(C.DECIMATION, device=self.device).view(-1, 1, 1)
        use_current = substeps >= self._delay_steps.view(1, -1, 1)
        self._delayed_runtime_actions.copy_(torch.where(
            use_current, runtime.unsqueeze(0), self._last_runtime_actions.unsqueeze(0)
        ))
        self._last_runtime_actions.copy_(runtime)
        if self.cfg.randomize_control:
            self._joint_injection.uniform_(*self.cfg.joint_injection_range)
            self._joint_injection.mul_(self._effort_limits)
        else:
            self._joint_injection.zero_()
        self._control_substep = 0

    def _apply_action(self):
        runtime_action = self._delayed_runtime_actions[self._control_substep]
        canonical_action = gather_canonical(runtime_action, self._canonical_to_runtime)
        target = self._default_pos + C.ACTION_SCALE * canonical_action
        self._joint_target.copy_(target)
        pos = gather_canonical(self._robot.data.joint_pos, self._canonical_to_runtime)
        vel = gather_canonical(self._robot.data.joint_vel, self._canonical_to_runtime)
        leg = self._leg_canonical
        efforts = compute_leg_efforts(
            target[:, leg], pos[:, leg], vel[:, leg],
            kp=self._kp[:, leg], kd=self._kd[:, leg],
            kp_factors=self._kp_factors[:, leg], kd_factors=self._kd_factors[:, leg],
            actuation_offset=self._actuation_offset[:, leg] + self._joint_injection[:, leg],
            effort_limits=self._effort_limits[:, leg],
        )
        self._applied_torques.zero_()
        self._applied_torques[:, leg] = efforts
        self._robot.set_joint_effort_target(efforts, joint_ids=self._leg_runtime)
        upper_target = target[:, self._upper_canonical]
        upper_target = torch.clamp(
            upper_target, self._hard_lower[:, self._upper_canonical], self._hard_upper[:, self._upper_canonical]
        )
        self._robot.set_joint_position_target(upper_target, joint_ids=self._upper_runtime)
        self._control_substep = (self._control_substep + 1) % C.DECIMATION

    def _imu(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self._robot.data.body_quat_w[:, self._imu_id]
        angular = math_utils.quat_apply_inverse(quat, self._robot.data.body_ang_vel_w[:, self._imu_id])
        gravity = math_utils.quat_apply_inverse(quat, self._gravity_w)
        return angular, gravity

    def _actor_frame(self, noisy: bool) -> torch.Tensor:
        pos = gather_canonical(self._robot.data.joint_pos, self._canonical_to_runtime)
        vel = gather_canonical(self._robot.data.joint_vel, self._canonical_to_runtime)
        angular, gravity = self._imu()
        commands = torch.cat((self._commands[:, :3], self._commands[:, 4:5]), dim=-1)
        frame = assemble_actor_frame(commands, angular, gravity, pos, self._default_pos, vel, self._actions)
        if noisy and self.cfg.add_noise:
            frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * self._noise
        self._assert_finite(frame, "actor observation")
        return frame

    def _critic(self) -> torch.Tensor:
        frame = self._actor_frame(noisy=False)
        critic = torch.cat((frame, self._robot.data.root_link_lin_vel_b * C.LIN_VEL_SCALE), dim=-1)
        if critic.shape[-1] != C.num_critic_obs():
            raise RuntimeError("ELF3 critic observation width mismatch")
        return critic

    def _snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "policy": torch.clamp(self._history.flatten(1), -self.cfg.clip_observations, self.cfg.clip_observations),
            "critic": torch.clamp(self._critic(), -self.cfg.clip_observations, self.cfg.clip_observations),
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self._last_history_step != self.common_step_counter:
            reset_ids = self._history_reset_ids
            self._history.copy_(shift_history(self._history, self._actor_frame(noisy=True), reset_ids))
            self._history_reset_ids = None
            self._last_history_step = self.common_step_counter
        return self._snapshot()

    def _resample_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        if self.cfg.use_random_commands:
            modes = curriculum.sample_modes(torch.rand(count, device=self.device))
            commands = curriculum.build_commands(
                modes=modes, velocity_draws=torch.rand(count, 3, device=self.device),
                height_draws=torch.rand(count, device=self.device), stage=self._stage,
                crouch_min_height=self.cfg.crouch_min_height,
                crouch_focus_max_height=self.cfg.crouch_focus_max_height,
            )
        else:
            modes = torch.full((count,), curriculum.HIGH_STAND, device=self.device, dtype=torch.long)
            commands = torch.zeros(count, 5, device=self.device)
            commands[:, 4] = self._stage.walk_height
        self._modes[env_ids] = modes
        self._commands[env_ids] = commands

    def _maybe_resample_commands(self):
        due = (self.episode_length_buf > 0) & (self.episode_length_buf % self._command_steps == 0)
        self._resample_commands(due.nonzero().flatten())

    def _body_points(self, points: torch.Tensor) -> torch.Tensor:
        root_pos = self._robot.data.root_link_pos_w
        root_quat = self._robot.data.root_link_quat_w
        return math_utils.quat_apply_inverse(root_quat.unsqueeze(1).expand(-1, points.shape[1], -1), points - root_pos.unsqueeze(1))

    def _body_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        root_vel = self._robot.data.root_link_lin_vel_w
        root_quat = self._robot.data.root_link_quat_w
        return math_utils.quat_apply_inverse(root_quat.unsqueeze(1).expand(-1, velocity.shape[1], -1), velocity - root_vel.unsqueeze(1))

    def _get_rewards(self) -> torch.Tensor:
        pos = gather_canonical(self._robot.data.joint_pos, self._canonical_to_runtime)
        vel = gather_canonical(self._robot.data.joint_vel, self._canonical_to_runtime)
        base_lin = self._robot.data.root_link_lin_vel_b
        base_ang = self._robot.data.root_link_ang_vel_b
        gravity = self._robot.data.projected_gravity_b
        root_height = self._robot.data.root_link_pos_w[:, 2]
        feet_pos = self._robot.data.body_pos_w[:, self._feet_ids]
        feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids]
        feet_height = feet_pos[..., 2]
        sole = virtual_sole_corners(feet_pos, self._robot.data.body_quat_w[:, self._feet_ids])
        sole_variance = sole[..., 2].var(dim=-1, unbiased=False)
        sole_distances = torch.linalg.vector_norm(sole[:, 0] - sole[:, 1], dim=-1)
        forces = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids]
        self._contact_forces.copy_(forces)
        contacts = torch.linalg.vector_norm(forces, dim=-1) > 1.0
        self._contact_filter.copy_(contacts | self._last_contacts)
        self._first_contacts.copy_((self._air_time >= self.step_dt) & self._contact_filter)
        self._air_time.add_(self.step_dt)
        feet_b = self._body_points(feet_pos)
        knees_b = self._body_points(self._robot.data.body_pos_w[:, self._knee_body_ids])
        feet_vel_b = self._body_velocity(feet_vel)
        foot_distance = torch.abs(feet_b[:, 0, 1] - feet_b[:, 1, 1])
        knee_distance = torch.abs(knees_b[:, 0, 1] - knees_b[:, 2, 1]) + torch.abs(knees_b[:, 1, 1] - knees_b[:, 3, 1])
        high = curriculum.high_mask(self._modes)
        stand = curriculum.stand_mask(self._modes)
        leg = C.LOWER_DOF_INDICES
        force_z = forces[..., 2]
        raw = {
            "tracking_x_vel": R.tracking_x_vel(self._commands, base_lin, self.cfg.tracking_sigma, self._modes),
            "tracking_y_vel": R.tracking_y_vel(self._commands, base_lin, self.cfg.tracking_sigma, self._modes),
            "tracking_ang_vel": R.tracking_ang_vel(self._commands, base_ang, self.cfg.tracking_sigma, self._modes),
            "tracking_base_height": R.tracking_base_height(root_height=root_height, feet_height=feet_height, commanded_height=self._commands[:, 4], ankle_sole_distance=self.cfg.ankle_sole_distance, modes=self._modes),
            "lin_vel_z": R.lin_vel_z(base_lin, high),
            "ang_vel_xy": R.ang_vel_xy(base_ang),
            "orientation": R.orientation(gravity),
            "action_rate": R.action_rate(self._previous_actions, self._actions),
            "deviation_hip_joint": R.deviation_hip_joint(pos, self._default_pos, _indices(HIP_JOINT_NAMES), high),
            "deviation_ankle_joint": R.deviation_ankle_joint(pos, self._default_pos, _indices(ANKLE_ROLL_JOINT_NAMES), high),
            "deviation_knee_joint": R.deviation_knee_joint(dof_pos=pos, joint_lower_limits=self._hard_lower, joint_upper_limits=self._hard_upper, knee_joint_indices=_indices(KNEE_JOINT_NAMES), root_height=root_height, commanded_height=self._commands[:, 4]),
            "dof_acc": R.dof_acc(vel, self._last_dof_vel, self.step_dt),
            "dof_pos_limits": R.dof_pos_limits(pos, self._soft_lower, self._soft_upper),
            "feet_air_time": R.feet_air_time(self._air_time, self._first_contacts, torch.linalg.vector_norm(self._commands[:, :3], dim=-1)),
            "feet_clearance": R.feet_clearance(feet_height, feet_vel_b[..., :2], self.cfg.clearance_height_target, high),
            "feet_distance_lateral": R.feet_distance_lateral(foot_distance, self.cfg.least_feet_distance_lateral, self.cfg.most_feet_distance_lateral),
            "knee_distance_lateral": R.knee_distance_lateral(knee_distance, self.cfg.least_knee_distance_lateral, self.cfg.most_knee_distance_lateral),
            "feet_ground_parallel": R.feet_ground_parallel(sole_variance, self._contact_filter),
            "feet_parallel": R.feet_parallel(sole_distances),
            "smoothness": R.smoothness(self._actions, self._previous_actions, self._second_previous_actions),
            "joint_power": R.joint_power(vel, self._applied_torques, self._commands),
            "feet_stumble": R.feet_stumble(forces),
            "torques": R.torques(self._applied_torques, self._kp.expand_as(pos), leg),
            "dof_vel": R.dof_vel(vel, leg),
            "dof_vel_limits": R.dof_vel_limits(vel, self._velocity_limits, self.cfg.soft_dof_vel_limit, leg),
            "torque_limits": R.torque_limits(self._applied_torques, self._effort_limits, self.cfg.soft_torque_limit, leg),
            "no_fly": R.no_fly(force_z, stand),
            "joint_tracking_error": R.joint_tracking_error(self._joint_target, pos, leg),
            "feet_slip": R.feet_slip(feet_vel[..., :2], force_z),
            "feet_contact_forces": R.feet_contact_forces(forces, self.cfg.max_contact_force),
            "contact_momentum": R.contact_momentum(feet_vel[..., 2], force_z),
            "action_vanish": R.action_vanish(self._actions, self._action_min, self._action_max),
            "stand_still": R.stand_still(force_z, high),
        }
        for name, value in raw.items():
            self._assert_finite(value, f"raw reward {name}")
        scaled = R.scale_reward_terms(raw, self.cfg.reward_scales, C.policy_dt())
        self.last_raw_reward_terms = raw
        self.last_scaled_reward_terms = scaled
        for name in R.REWARD_NAMES:
            self._episode_sums[name].add_(scaled[name])
        total = R.sum_reward_terms(scaled)
        self._second_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._last_dof_vel.copy_(vel)
        self._last_contacts.copy_(contacts)
        self._air_time.mul_((~self._contact_filter).to(self._air_time.dtype))
        self._assert_finite(total, "reward")
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_resample_commands()
        torso_history = self._contact_sensor.data.net_forces_w_history[:, :, self._torso_contact_ids]
        torso = torch.any(
            torch.amax(torch.linalg.vector_norm(torso_history, dim=-1), dim=1) > self.cfg.torso_contact_threshold,
            dim=-1,
        )
        gravity = torch.linalg.vector_norm(self._robot.data.projected_gravity_b[:, :2], dim=-1) > self.cfg.gravity_xy_termination
        failure = gravity | (torso & self.cfg.terminate_on_torso_contact)
        terminated, truncated, time_outs = classify_dones(failure, self.episode_length_buf, self.max_episode_length)
        terminal = self._critic().detach().clone()
        self.extras["terminal_critic_obs"] = terminal
        self.extras["terminal_critic_obs_mask"] = (terminated | truncated).detach().clone()
        self.extras["time_outs"] = time_outs.detach().clone()
        return terminated, truncated

    def _randomize_control(self, env_ids: torch.Tensor):
        if self.cfg.randomize_control:
            shape = (env_ids.numel(), C.NUM_ROBOT_DOFS)
            draws = {name: torch.rand(shape, device=self.device) for name in ("kp", "kd", "offset")}
            offset_fraction = torch.zeros_like(self._actuation_offset)
            apply_control_randomization(
                self._kp_factors, self._kd_factors, offset_fraction, env_ids,
                draws=draws, kp_range=self.cfg.kp_factor_range, kd_range=self.cfg.kd_factor_range,
                offset_range=self.cfg.actuation_offset_range,
            )
            self._actuation_offset[env_ids] = offset_fraction[env_ids] * self._effort_limits[env_ids]
        else:
            self._kp_factors[env_ids] = 1.0
            self._kd_factors[env_ids] = 1.0
            self._actuation_offset[env_ids] = 0.0
        self._joint_injection[env_ids] = 0.0
        self._write_upper_gains(env_ids)

    def _write_upper_gains(self, env_ids: torch.Tensor):
        p_runtime = scatter_canonical(self._kp * self._kp_factors, self._canonical_to_runtime, C.NUM_ROBOT_DOFS)
        d_runtime = scatter_canonical(self._kd * self._kd_factors, self._canonical_to_runtime, C.NUM_ROBOT_DOFS)
        stiffness = p_runtime[env_ids][:, self._upper_runtime]
        damping = d_runtime[env_ids][:, self._upper_runtime]
        actuator = self._robot.actuators["upper_body"]
        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        self._robot.write_joint_stiffness_to_sim(stiffness, joint_ids=self._upper_runtime, env_ids=env_ids)
        self._robot.write_joint_damping_to_sim(damping, joint_ids=self._upper_runtime, env_ids=env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        lengths = torch.clamp(self.episode_length_buf[ids], min=1)
        log = {f"Episode_Reward/{name}": torch.mean(values[ids] / lengths) for name, values in self._episode_sums.items()}
        self._robot.reset(ids)
        super()._reset_idx(ids)
        joint_pos = self._robot.data.default_joint_pos[ids].clone()
        joint_vel = self._robot.data.default_joint_vel[ids].clone().zero_()
        if self.cfg.randomize_initial_state:
            joint_pos.mul_(torch.empty_like(joint_pos).uniform_(*self.cfg.initial_joint_pos_scale))
            joint_pos.add_(torch.empty_like(joint_pos).uniform_(*self.cfg.initial_joint_pos_offset))
        limits = self._robot.data.soft_joint_pos_limits[ids]
        joint_pos.clamp_(limits[..., 0], limits[..., 1])
        root = self._robot.data.default_root_state[ids].clone()
        root[:, :3] += self._terrain.env_origins[ids]
        if self.cfg.randomize_initial_state:
            root[:, 7:13].uniform_(*self.cfg.initial_root_velocity_range)
        else:
            root[:, 7:13] = 0.0
        self._robot.write_root_pose_to_sim(root[:, :7], ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)
        self._randomize_control(ids)
        for tensor in (
            self._actions, self._previous_actions, self._second_previous_actions,
            self._last_runtime_actions, self._upper_actions, self._upper_target,
            self._upper_delta, self._applied_torques, self._last_dof_vel,
            self._air_time, self._contact_forces,
        ):
            tensor[ids] = 0.0
        self._delayed_runtime_actions[:, ids] = 0.0
        self._delay_steps[ids] = 0
        self._last_contacts[ids] = False
        self._contact_filter[ids] = False
        self._first_contacts[ids] = False
        self._joint_target[ids] = self._default_pos[ids]
        self._history[ids] = 0.0
        self._history_reset_ids = ids
        self._resample_commands(ids)
        for values in self._episode_sums.values():
            values[ids] = 0.0
        self.extras["log"] = log

    @staticmethod
    def _assert_finite(tensor: torch.Tensor, name: str):
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).reshape(tensor.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
            raise RuntimeError(f"non-finite ELF3 {name} in environments {bad}")

    @property
    def canonical_to_runtime_dof_indices(self) -> torch.Tensor:
        return self._canonical_to_runtime

    @property
    def applied_torques_canonical(self) -> torch.Tensor:
        return self._applied_torques

    @property
    def effort_limits_canonical(self) -> torch.Tensor:
        return self._effort_limits

    @property
    def reward_term_names(self) -> tuple[str, ...]:
        return R.REWARD_NAMES

    @property
    def reward_scales(self) -> dict[str, float]:
        return dict(self.cfg.reward_scales)
