"""Controller state for the HOMIE crouch-and-walk policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.mod_api import (
    ResourceHandle,
    RobotControlState,
    StateBehavior,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)
from bxi_example_py_elf3.policies import HomiePolicy, PolicySafetyError

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class HomieState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    """Adapt the shared controller inputs to HOMIE's sim2real contract."""

    HEIGHT_RATE_MPS = 0.35

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HomiePolicy],
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self._height_command = HomiePolicy.HEIGHT_MAX_M
        self._safety_frame: MotorFrame | None = None
        self._failure_latched = False

    @property
    def policy(self) -> HomiePolicy:
        return self._policy.get()

    @property
    def height_command(self) -> float:
        return self._height_command

    @property
    def failure_latched(self) -> bool:
        return self._failure_latched

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        del from_state
        if self._failure_latched:
            self._apply_frame(ctx, self._zero_torque_frame(ctx))
            return
        self._height_command = HomiePolicy.HEIGHT_MAX_M
        self.policy.set_height_command(self._height_command)
        try:
            ctx.preheat_model(self.policy, command=self.get_cmd_vel(ctx))
        except Exception as exc:
            self._apply_frame(ctx, self._latch_failure(ctx, exc, request=False))

    def on_enter(self, ctx: RobotControlContext) -> None:
        if self._failure_latched:
            self._request_zero_torque(ctx)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        if self._failure_latched:
            return self._zero_torque_frame(ctx)
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    @staticmethod
    def _height_rate(ctx: RobotControlContext) -> float:
        rate = float(ctx.current_raw_height_rate)
        if not np.isfinite(rate):
            raise PolicySafetyError("HOMIE height rate must be finite")
        return float(np.clip(rate, -1.0, 1.0))

    def _zero_torque_frame(self, ctx: RobotControlContext) -> MotorFrame:
        frame = self._safety_frame
        if frame is None or frame.layout != ctx.robot_layout:
            frame = MotorFrame.empty(ctx.robot_layout)
            self._safety_frame = frame
        np.copyto(frame.qpos, ctx.robot_joints.position, casting="same_kind")
        frame.kp.fill(0.0)
        frame.kd.fill(0.0)
        frame.vel.fill(0.0)
        frame.torque.fill(0.0)
        return frame

    def _request_zero_torque(
        self,
        ctx: RobotControlContext,
    ) -> MotorFrame:
        ctx.current_cmd_vel.fill(0.0)
        ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
        return self._zero_torque_frame(ctx)

    def _latch_failure(
        self,
        ctx: RobotControlContext,
        error: BaseException,
        *,
        request: bool = True,
    ) -> MotorFrame:
        if not self._failure_latched:
            self._failure_latched = True
            if self._logger is not None:
                self.logger.error(f"HOMIE safety stop: {error}")
        ctx.current_cmd_vel.fill(0.0)
        if request:
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
        return self._zero_torque_frame(ctx)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            return self._request_zero_torque(ctx)
        if self._failure_latched:
            return self._request_zero_torque(ctx)
        try:
            self.get_cmd_vel(ctx)
            height = self._height_command
            if advance:
                if not np.isfinite(dt) or dt < 0.0:
                    raise PolicySafetyError(
                        "HOMIE state dt must be finite and non-negative"
                    )
                height = float(
                    np.clip(
                        height + self._height_rate(ctx) * self.HEIGHT_RATE_MPS * dt,
                        HomiePolicy.HEIGHT_MIN_M,
                        HomiePolicy.HEIGHT_MAX_M,
                    )
                )
            output = self.policy.step(
                ctx.inference_frame,
                dt,
                height_command=height,
                advance=advance,
            )
        except Exception as exc:
            return self._latch_failure(ctx, exc)
        if advance:
            self._height_command = height
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))


HomieLocomotionState = HomieState


__all__ = ["HomieLocomotionState", "HomieState"]
