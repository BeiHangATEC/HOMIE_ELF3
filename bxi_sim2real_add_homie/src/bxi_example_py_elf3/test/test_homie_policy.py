from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from bxi_example_py_elf3.framework.inference import InferenceFrame
from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointCommandResolver,
    JointDefault,
    JointLayout,
    JointStateView,
)
from bxi_example_py_elf3.framework.mod_api import (
    EntryFrameProvider,
    MotorFrame,
    RobotControlState,
    RunningFrameProvider,
)
from bxi_example_py_elf3.framework.platform.api import RobotObservation
from bxi_example_py_elf3.framework.runtime.state_machine import RobotStateMachine
from bxi_example_py_elf3.policies import (
    ELF3_POLICY_JOINTS,
    HOMIE_ACTION_JOINT_NAMES,
    HOMIE_OBSERVATION_JOINT_NAMES,
    HOMIE_ONNX_METADATA,
    HOMIE_PARAMETERS,
    HomiePolicy,
    PolicySafetyError,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = PACKAGE_ROOT / "mods" / "com.bxi.homie"


class FakeBackend:
    backend_name = "onnxruntime"
    input_names = ("obs",)
    output_names = ("actions",)

    def __init__(
        self,
        *,
        metadata=None,
        input_shape=(None, 468),
        output_shape=(None, 12),
    ) -> None:
        self.metadata = dict(HOMIE_ONNX_METADATA if metadata is None else metadata)
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)
        self.next_action = np.zeros((1, 12), dtype=np.float32)
        self.failure: BaseException | None = None
        self.calls: list[np.ndarray] = []
        self.closed = False

    def input_shape(self, name):
        assert name == "obs"
        return self._input_shape

    def output_shape(self, name):
        assert name == "actions"
        return self._output_shape

    def run(self, inputs):
        assert set(inputs) == {"obs"}
        self.calls.append(np.array(inputs["obs"], copy=True))
        if self.failure is not None:
            raise self.failure
        return {"actions": np.array(self.next_action, copy=True)}

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.options = SimpleNamespace(warmup_runs=1)
        self.opened = None

    def open_backend(self, spec, *, backend):
        self.opened = (spec, backend)
        return self.backend


def make_policy(backend: FakeBackend | None = None):
    selected = backend or FakeBackend()
    runtime = FakeRuntime(selected)
    policy = HomiePolicy("unused-homie.onnx", runtime=runtime)
    return policy, selected, runtime


def make_frame(
    *,
    command=(0.0, 0.0, 0.0),
    offsets=None,
    velocities=None,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
):
    offsets = np.zeros(28, dtype=np.float64) if offsets is None else offsets
    velocities = (
        np.zeros(28, dtype=np.float64) if velocities is None else velocities
    )
    default_by_name = dict(
        zip(ELF3_POLICY_JOINTS.names, HOMIE_PARAMETERS.default_position)
    )
    offset_by_name = dict(zip(HOMIE_OBSERVATION_JOINT_NAMES, offsets))
    velocity_by_name = dict(zip(HOMIE_OBSERVATION_JOINT_NAMES, velocities))

    # Reverse the controller layout so the test exercises semantic binding
    # instead of accidentally succeeding through matching numeric indices.
    source_layout = JointLayout(
        tuple(reversed(ELF3_POLICY_JOINTS.names)),
        label="test controller order",
    )
    position = np.asarray(
        [
            default_by_name[name] + offset_by_name.get(name, 0.0)
            for name in source_layout.names
        ],
        dtype=np.float64,
    )
    velocity = np.asarray(
        [velocity_by_name.get(name, 0.0) for name in source_layout.names],
        dtype=np.float64,
    )
    joints = JointStateView(source_layout, position, velocity)
    frame = InferenceFrame(
        joints=joints,
        quat_wxyz=np.asarray(quaternion, dtype=np.float64),
        angular_velocity=np.asarray(angular_velocity, dtype=np.float64),
        command=np.asarray(command, dtype=np.float32),
    )
    return frame


def test_uses_onnx_runtime_and_validates_the_declared_model_contract():
    policy, backend, runtime = make_policy()

    spec, selected_backend = runtime.opened
    assert selected_backend == "onnxruntime"
    assert spec.input_names == ("obs",)
    assert spec.output_names == ("actions",)
    assert len(spec.artifacts) == 1
    assert spec.artifacts[0].backend == "onnxruntime"
    assert backend.calls == []

    policy.close()
    assert backend.closed


def test_packaged_onnx_matches_the_declared_model_contract():
    onnxruntime = pytest.importorskip("onnxruntime")
    model_path = MOD_ROOT / "assets" / "homie_elf3.onnx"

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "ae581047fcb8bdb959c989f83c5169724a924bf3fb1c57b1f977817f3f10705b"
    )

    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=("CPUExecutionProvider",),
    )

    assert session.get_modelmeta().custom_metadata_map == HOMIE_ONNX_METADATA
    assert [(value.name, value.shape) for value in session.get_inputs()] == [
        ("obs", ["batch", 468])
    ]
    assert [(value.name, value.shape) for value in session.get_outputs()] == [
        ("actions", ["batch", 12])
    ]
    outputs = session.run(
        ("actions",),
        {"obs": np.zeros((1, 468), dtype=np.float32)},
    )
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 12)
    assert np.isfinite(outputs[0]).all()


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (
            FakeBackend(input_shape=(1, 467)),
            "input shape",
        ),
        (
            FakeBackend(output_shape=(1, 13)),
            "output shape",
        ),
        (
            FakeBackend(
                metadata={
                    key: value
                    for key, value in HOMIE_ONNX_METADATA.items()
                    if key != "source_checkpoint_sha256"
                }
            ),
            "missing required keys",
        ),
        (
            FakeBackend(
                metadata={**HOMIE_ONNX_METADATA, "history_length": "5"}
            ),
            "metadata mismatch",
        ),
    ],
)
def test_rejects_incompatible_backend_contracts_and_closes_them(backend, message):
    with pytest.raises(ValueError, match=message):
        make_policy(backend)
    assert backend.closed


def test_builds_exact_oldest_to_newest_observation_and_named_full_target():
    policy, backend, _runtime = make_policy()
    offsets = np.arange(1, 29, dtype=np.float64) / 100.0
    velocities = np.arange(1, 29, dtype=np.float64)
    frame = make_frame(
        command=(0.25, -0.5, 0.4),
        offsets=offsets,
        velocities=velocities,
        angular_velocity=(2.0, -4.0, 6.0),
    )
    policy.set_height_command(0.65)
    policy.reset(frame)
    backend.next_action[0] = np.arange(-6, 6, dtype=np.float32)

    output = policy.step(frame, 0.02)
    history = backend.calls[-1].reshape(6, 78)
    expected = np.concatenate(
        (
            np.asarray((0.5, -1.0, 0.2, 0.65), dtype=np.float32),
            np.asarray((1.0, -2.0, 3.0), dtype=np.float32),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            offsets.astype(np.float32),
            (velocities * 0.05).astype(np.float32),
            np.zeros(12, dtype=np.float32),
        )
    )
    assert history.shape == (6, 78)
    np.testing.assert_array_equal(history[:5], np.zeros((5, 78), dtype=np.float32))
    np.testing.assert_allclose(history[-1], expected, atol=2.0e-7)

    assert output.joints.layout == ELF3_POLICY_JOINTS
    expected_target = HOMIE_PARAMETERS.default_position.copy()
    for action, name in zip(backend.next_action[0], HOMIE_ACTION_JOINT_NAMES):
        expected_target[ELF3_POLICY_JOINTS.index(name)] += action * 0.25
    expected_target[ELF3_POLICY_JOINTS.index("waist_x_joint")] = 0.0
    np.testing.assert_allclose(output.joints.position, expected_target)
    np.testing.assert_array_equal(output.joints.kp, HOMIE_PARAMETERS.kp)
    np.testing.assert_array_equal(output.joints.kd, HOMIE_PARAMETERS.kd)

    action_indices = [ELF3_POLICY_JOINTS.index(name) for name in HOMIE_ACTION_JOINT_NAMES]
    assert np.all(HOMIE_PARAMETERS.action_scale[action_indices] == 0.25)
    non_action_indices = sorted(set(range(29)) - set(action_indices))
    assert np.all(HOMIE_PARAMETERS.action_scale[non_action_indices] == 0.0)


def test_clips_actions_to_the_training_range_before_scaling_and_feedback():
    policy, backend, _runtime = make_policy()
    frame = make_frame()
    policy.reset(frame)
    backend.next_action[0] = np.asarray(
        (-101.0, 101.0, -500.0, 500.0, -100.0, 100.0) * 2,
        dtype=np.float32,
    )

    output = policy.step(frame, 0.02)

    clipped = np.clip(backend.next_action[0], -100.0, 100.0)
    for action, name in zip(clipped, HOMIE_ACTION_JOINT_NAMES):
        index = ELF3_POLICY_JOINTS.index(name)
        assert output.joints.position[index] == pytest.approx(
            HOMIE_PARAMETERS.default_position[index] + action * 0.25
        )
    policy.step(frame, 0.02, advance=False)
    newest = backend.calls[-1].reshape(6, 78)[-1]
    np.testing.assert_array_equal(newest[66:78], clipped)


def test_named_output_resolves_to_full_robot_with_explicit_head_defaults():
    policy, _backend, _runtime = make_policy()
    source = MotorFrame.create(
        policy.output.joints.layout,
        policy.output.joints.position,
        policy.output.joints.kp,
        policy.output.joints.kd,
    )
    robot_layout = JointLayout(
        ELF3_POLICY_JOINTS.names + ("head_z_joint", "head_y_joint"),
        label="ELF3 31-joint robot",
    )
    defaults = JointCommandDefaults(
        {
            "head_z_joint": JointDefault(position=0.12, kp=30.0, kd=1.5),
            "head_y_joint": JointDefault(position=-0.08, kp=40.0, kd=2.0),
        }
    )
    resolved = MotorFrame.empty(robot_layout)

    JointCommandResolver(robot_layout, defaults).resolve_into(source, resolved)

    assert resolved.layout.dof_num == 31
    np.testing.assert_array_equal(resolved.qpos[:29], source.qpos)
    np.testing.assert_array_equal(resolved.kp[:29], source.kp)
    np.testing.assert_array_equal(resolved.kd[:29], source.kd)
    np.testing.assert_allclose(resolved.qpos[29:], (0.12, -0.08))
    np.testing.assert_allclose(resolved.kp[29:], (30.0, 40.0))
    np.testing.assert_allclose(resolved.kd[29:], (1.5, 2.0))


def test_preview_does_not_advance_history_previous_action_or_height():
    policy, backend, _runtime = make_policy()
    initial = make_frame(command=(0.0, 0.0, 0.0))
    policy.reset(initial)

    backend.next_action.fill(1.0)
    first = make_frame(command=(0.1, 0.0, 0.0))
    policy.step(first, 0.02, advance=True)

    backend.next_action.fill(2.0)
    preview = make_frame(command=(0.2, 0.0, 0.0))
    policy.step(preview, 0.02, height_command=0.50, advance=False)
    preview_input = backend.calls[-1].copy()
    assert policy.height_command == HomiePolicy.HEIGHT_MAX_M

    policy.step(preview, 0.02, height_command=0.50, advance=False)
    np.testing.assert_array_equal(backend.calls[-1], preview_input)

    policy.step(preview, 0.02, height_command=0.50, advance=True)
    np.testing.assert_array_equal(backend.calls[-1], preview_input)
    assert policy.height_command == 0.50

    backend.next_action.fill(3.0)
    policy.step(preview, 0.02, advance=False)
    newest = backend.calls[-1].reshape(6, 78)[-1]
    np.testing.assert_array_equal(newest[66:78], np.full(12, 2.0))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.command.__setitem__(0, np.nan),
        lambda frame: frame.command.__setitem__(0, 0.51),
        lambda frame: frame.angular_velocity.__setitem__(1, np.inf),
        lambda frame: frame.quat_wxyz.fill(0.0),
        lambda frame: frame.joints.position.__setitem__(0, np.nan),
        lambda frame: frame.joints.velocity.__setitem__(0, np.inf),
    ],
)
def test_rejects_non_finite_or_out_of_contract_inputs_before_inference(mutate):
    policy, backend, _runtime = make_policy()
    frame = make_frame()
    mutate(frame)

    with pytest.raises(PolicySafetyError):
        policy.step(frame, 0.02)
    assert backend.calls == []


@pytest.mark.parametrize("height", [0.39, 1.02, np.nan, np.inf])
def test_rejects_height_outside_the_training_contract(height):
    policy, backend, _runtime = make_policy()

    with pytest.raises(PolicySafetyError):
        policy.step(make_frame(), 0.02, height_command=height)
    assert backend.calls == []


@pytest.mark.parametrize("height", [0.40, 1.01])
def test_accepts_height_at_the_training_contract_boundaries(height):
    policy, backend, _runtime = make_policy()

    policy.step(make_frame(), 0.02, height_command=height)

    assert len(backend.calls) == 1
    assert backend.calls[-1].reshape(6, 78)[-1, 3] == pytest.approx(height)


def test_rejects_invalid_outputs_and_wraps_backend_failures():
    policy, backend, _runtime = make_policy()
    frame = make_frame()

    backend.next_action[0, 3] = np.nan
    with pytest.raises(PolicySafetyError, match="finite floats"):
        policy.step(frame, 0.02)

    backend.next_action.fill(0.0)
    backend.failure = RuntimeError("backend stopped")
    with pytest.raises(PolicySafetyError, match="inference failed"):
        policy.step(frame, 0.02)


def _load_state_module():
    spec = importlib.util.spec_from_file_location("_test_homie_state", MOD_ROOT / "state.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHandle:
    def __init__(self, policy) -> None:
        self.policy = policy

    def get(self):
        return self.policy

    @property
    def status(self):
        return "ready"


class FixedFrameState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(self, name, state_id, frame):
        super().__init__(name, state_id)
        self.frame = frame

    def get_entry_frame(self, ctx):
        return self.frame

    def sample_running_frame(self, ctx, dt, *, advance):
        return self.frame

    def on_update(self, ctx, dt):
        self._apply_frame(ctx, self.frame)


class SilentLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class FakeContext:
    def __init__(self, frame) -> None:
        self.inference_frame = frame
        self.robot_layout = frame.joints.layout
        self.robot_joints = frame.joints
        self.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.current_raw_height_rate = 0.0
        self.current_cmd_vel = frame.command
        self.current_quat_xyzw = np.asarray((0.0, 0.0, 0.0, 1.0))
        self.speed_profiles = {
            "homie": {
                "vx_scale": 0.5,
                "vx_min": -0.5,
                "vx_max": 0.5,
                "vy_scale": 0.5,
                "vy_min": -0.5,
                "vy_max": 0.5,
                "yaw_scale": 0.5,
                "yaw_min": -0.5,
                "yaw_max": 0.5,
            }
        }
        self.unsafe = False
        self.requests = []
        self.motor_target = None

    def preheat_model(self, model, command=None):
        if command is not None:
            np.copyto(self.current_cmd_vel, command)
        model.reset(self.inference_frame)
        model.step(self.inference_frame, 0.02, advance=False)

    def is_orientation_unsafe(self, quaternion):
        del quaternion
        return self.unsafe

    def request_state(self, state_name, *, trigger, **kwargs):
        self.requests.append((state_name, trigger, kwargs))
        return True

    def set_motor_target(self, frame):
        self.motor_target = frame


def test_state_integrates_trigger_height_only_when_advancing_and_clamps():
    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    context = FakeContext(make_frame())
    state = state_module.HomieState("com.bxi.homie/homie", 20, FakeHandle(policy))
    state.speed_profile_name = "homie"

    state.on_prepare(context, object())
    assert state.height_command == 1.01
    assert policy.height_command == 1.01
    assert len(backend.calls) == 1

    context.current_raw_height_rate = -1.0
    state.sample_running_frame(context, 1.0, advance=False)
    assert state.height_command == 1.01
    assert backend.calls[-1].reshape(6, 78)[-1, 3] == pytest.approx(1.01)

    state.sample_running_frame(context, 1.0, advance=True)
    assert state.height_command == pytest.approx(0.66)
    assert backend.calls[-1].reshape(6, 78)[-1, 3] == pytest.approx(0.66)

    state.sample_running_frame(context, 10.0, advance=True)
    assert state.height_command == 0.40
    context.current_raw_height_rate = 1.0
    state.sample_running_frame(context, 10.0, advance=True)
    assert state.height_command == 1.01


def test_state_routes_unsafe_orientation_to_zero_torque_without_inference():
    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    context = FakeContext(make_frame())
    state = state_module.HomieState("com.bxi.homie/homie", 20, FakeHandle(policy))
    state.speed_profile_name = "homie"
    state.on_prepare(context, object())
    call_count = len(backend.calls)
    context.unsafe = True

    state.on_update(context, 0.02)

    assert len(backend.calls) == call_count
    assert context.requests == [
        ("com.bxi.basic_actions/zero_torque", "safety", {})
    ]
    assert context.motor_target is not None
    np.testing.assert_array_equal(context.motor_target.kp, 0.0)
    np.testing.assert_array_equal(context.motor_target.kd, 0.0)


def test_state_preview_routes_unsafe_orientation_to_zero_torque_without_inference():
    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    context = FakeContext(make_frame())
    state = state_module.HomieState("com.bxi.homie/homie", 20, FakeHandle(policy))
    state.speed_profile_name = "homie"
    state.on_prepare(context, object())
    call_count = len(backend.calls)
    context.unsafe = True

    safe_frame = state.sample_running_frame(context, 0.02, advance=False)

    assert len(backend.calls) == call_count
    assert context.requests == [
        ("com.bxi.basic_actions/zero_torque", "safety", {})
    ]
    np.testing.assert_array_equal(safe_frame.kp, 0.0)
    np.testing.assert_array_equal(safe_frame.kd, 0.0)


@pytest.mark.parametrize("advance", [False, True])
def test_state_turns_inference_failures_into_zero_torque_for_all_sampling(advance):
    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    context = FakeContext(make_frame())
    state = state_module.HomieState("com.bxi.homie/homie", 20, FakeHandle(policy))
    state.speed_profile_name = "homie"
    state.on_prepare(context, object())
    original_height = state.height_command
    backend.failure = RuntimeError("inference unavailable")

    safe_frame = state.sample_running_frame(context, 0.5, advance=advance)

    assert state.height_command == original_height
    assert policy.height_command == original_height
    assert state.failure_latched
    assert context.requests[-1][:2] == (
        "com.bxi.basic_actions/zero_torque",
        "safety",
    )
    assert safe_frame.layout == context.robot_layout
    np.testing.assert_array_equal(safe_frame.qpos, context.robot_joints.position)
    np.testing.assert_array_equal(safe_frame.kp, 0.0)
    np.testing.assert_array_equal(safe_frame.kd, 0.0)


def test_state_latches_preheat_failure_and_requests_zero_torque_on_entry():
    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    context = FakeContext(make_frame())
    state = state_module.HomieState("com.bxi.homie/homie", 20, FakeHandle(policy))
    state.speed_profile_name = "homie"
    backend.failure = RuntimeError("preheat unavailable")

    state.on_prepare(context, object())

    assert state.failure_latched
    assert context.requests == []
    assert context.motor_target is not None
    np.testing.assert_array_equal(context.motor_target.kp, 0.0)
    state.on_enter(context)
    assert context.requests[-1][:2] == (
        "com.bxi.basic_actions/zero_torque",
        "safety",
    )


def _transition_framework(monkeypatch):
    ament_module = ModuleType("ament_index_python")
    packages_module = ModuleType("ament_index_python.packages")

    class PackageNotFoundError(Exception):
        pass

    packages_module.PackageNotFoundError = PackageNotFoundError
    packages_module.get_package_prefix = lambda package: "/unused"
    ament_module.packages = packages_module
    monkeypatch.setitem(sys.modules, "ament_index_python", ament_module)
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", packages_module)
    from bxi_example_py_elf3.framework.runtime.controller import RobotControlFramework

    state_module = _load_state_module()
    policy, backend, _runtime = make_policy()
    inference_frame = make_frame()
    layout = inference_frame.joints.layout
    source_frame = MotorFrame.create(
        layout,
        inference_frame.joints.position,
        np.full(layout.dof_num, 10.0, dtype=np.float32),
        np.full(layout.dof_num, 1.0, dtype=np.float32),
    )
    zero_frame = MotorFrame.create(
        layout,
        inference_frame.joints.position,
        np.zeros(layout.dof_num, dtype=np.float32),
        np.zeros(layout.dof_num, dtype=np.float32),
    )
    homie = state_module.HomieState("homie", 2, FakeHandle(policy))
    homie.speed_profile_name = "homie"
    states = {
        "source": FixedFrameState("source", 1, source_frame),
        "homie": homie,
        "com.bxi.basic_actions/zero_torque": FixedFrameState(
            "com.bxi.basic_actions/zero_torque",
            3,
            zero_frame,
        ),
    }
    config = {
        "initial_state": "source",
        "default_transition": "instant",
        "transition_profiles": {
            "instant": {"type": "instant"},
            "blend": {
                "type": "running_blend",
                "duration": 1.0,
                "curve": "linear",
                "sample_from": True,
                "sample_to": True,
                "advance_from": True,
                "advance_to": False,
            },
        },
        "states": {
            "source": {
                "transitions": {
                    "on_event": {
                        "start": {"to": "homie", "transition": "blend"}
                    }
                }
            },
            "homie": {
                "transitions": {
                    "on_event": {
                        "exit": {"to": "source", "transition": "blend"}
                    }
                }
            },
            "com.bxi.basic_actions/zero_torque": {},
        },
        "graph": {"validate": False},
    }

    framework = object.__new__(RobotControlFramework)
    framework._closed = False
    framework._logger = SilentLogger()
    framework._command_defaults = JointCommandDefaults()
    framework._robot_layout = None
    framework._robot_joints = None
    framework._inference_frame = None
    framework._command_resolver = None
    framework._resolved_motor_frame = None
    framework._last_motor_frame = None
    framework._direct_motor_layout = None
    framework._motor_target = None
    framework._pending_state_requests = []
    framework._initial_state_entered = False
    framework.current_quat_xyzw = np.zeros(4, dtype=np.float64)
    framework.current_quat_wxyz = np.zeros(4, dtype=np.float64)
    framework.current_omega = np.zeros(3, dtype=np.float64)
    framework.current_linear_acceleration = np.zeros(3, dtype=np.float64)
    framework.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
    framework.current_raw_height_rate = 0.0
    framework.current_cmd_vel = np.zeros(3, dtype=np.float32)
    framework.speed_profiles = {
        "homie": {
            "vx_scale": 0.5,
            "vx_min": -0.5,
            "vx_max": 0.5,
            "vy_scale": 0.5,
            "vy_min": -0.5,
            "vy_max": 0.5,
            "yaw_scale": 0.5,
            "yaw_min": -0.5,
            "yaw_max": 0.5,
        }
    }
    framework.loop_count = 0
    framework.state_machine = RobotStateMachine(
        framework,
        config,
        states,
        logger=framework._logger,
        enter_initial=False,
    )
    observation = RobotObservation(
        joints=inference_frame.joints,
        quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        quat_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        raw_cmd_vel=np.zeros(3, dtype=np.float32),
    )
    return framework, backend, observation


def test_replaced_running_transition_cannot_overwrite_safety_frame(monkeypatch):
    framework, backend, observation = _transition_framework(monkeypatch)
    first_frame = framework.update(observation, ("start",), 0.1)
    assert first_frame is not None
    assert framework.state_machine.in_transition
    backend.failure = RuntimeError("inference unavailable")

    safe_frame = framework.update(observation, (), 0.1)

    assert framework.current_state_name == "com.bxi.basic_actions/zero_torque"
    _assert_zero_torque_frame(safe_frame, observation.joints)


def _unsafe_observation(observation):
    half_sqrt_two = np.sqrt(0.5)
    return RobotObservation(
        joints=observation.joints,
        quat_xyzw=np.asarray(
            (half_sqrt_two, 0.0, 0.0, half_sqrt_two),
            dtype=np.float64,
        ),
        quat_wxyz=np.asarray(
            (half_sqrt_two, half_sqrt_two, 0.0, 0.0),
            dtype=np.float64,
        ),
        omega=observation.omega,
        raw_cmd_vel=observation.raw_cmd_vel,
    )


def _assert_zero_torque_frame(frame, joints):
    assert frame is not None
    np.testing.assert_array_equal(frame.qpos, joints.position)
    np.testing.assert_array_equal(frame.kp, 0.0)
    np.testing.assert_array_equal(frame.kd, 0.0)
    np.testing.assert_array_equal(frame.vel, 0.0)
    np.testing.assert_array_equal(frame.torque, 0.0)


def test_unsafe_orientation_as_homie_blend_target_outputs_zero_torque(monkeypatch):
    framework, backend, observation = _transition_framework(monkeypatch)

    safe_frame = framework.update(_unsafe_observation(observation), ("start",), 0.1)

    assert framework.current_state_name == "com.bxi.basic_actions/zero_torque"
    assert len(backend.calls) == 1
    _assert_zero_torque_frame(safe_frame, observation.joints)


def test_unsafe_orientation_as_homie_blend_source_outputs_zero_torque(monkeypatch):
    framework, _backend, observation = _transition_framework(monkeypatch)
    framework.update(observation, ("start",), 1.0)
    assert framework.current_state_name == "homie"

    safe_frame = framework.update(_unsafe_observation(observation), ("exit",), 0.1)

    assert framework.current_state_name == "com.bxi.basic_actions/zero_torque"
    _assert_zero_torque_frame(safe_frame, observation.joints)


def test_mod_manifest_declares_toggle_routes_bounds_and_safe_blends():
    manifest = yaml.safe_load((MOD_ROOT / "mod.yaml").read_text(encoding="utf-8"))

    assert manifest["id"] == "com.bxi.homie"
    assert manifest["requires"] == [
        {"id": "com.bxi.basic_actions", "version": ">=1,<2"}
    ]
    assert manifest["events"]["toggle"] == {"slot": "btn_10", "value": 13}
    assert manifest["runtime_requirements"]["python"] == [
        {"import": "onnxruntime"}
    ]
    profile = manifest["speed_profiles"]["homie"]
    assert profile == {
        "vx_scale": 0.5,
        "vx_min": -0.5,
        "vx_max": 0.5,
        "vy_scale": 0.5,
        "vy_min": -0.5,
        "vy_max": 0.5,
        "yaw_scale": 0.5,
        "yaw_min": -0.5,
        "yaw_max": 0.5,
    }
    blend = manifest["transition_profiles"]["safe_running_blend"]
    assert blend["type"] == "running_blend"
    assert blend["curve"] in {"linear", "smoothstep", "smootherstep"}
    assert blend["advance_to"] is False

    routes = {
        (route["from"], route["event"], route["to"]): route.get("transition")
        for route in manifest["routes"]
    }
    assert routes[
        ("com.bxi.basic_actions/normal", "toggle", "homie")
    ] == "safe_running_blend"
    assert routes[
        ("com.bxi.basic_actions/pd_brake", "toggle", "homie")
    ] == "safe_running_blend"
    assert routes[("homie", "toggle", "com.bxi.basic_actions/normal")] == (
        "safe_running_blend"
    )
    assert routes[
        ("homie", "com.bxi.basic_actions/normal", "com.bxi.basic_actions/normal")
    ] == "safe_running_blend"
    assert routes[
        (
            "homie",
            "com.bxi.basic_actions/pd_brake",
            "com.bxi.basic_actions/pd_brake",
        )
    ] == "safe_running_blend"
    assert routes[
        (
            "homie",
            "com.bxi.basic_actions/zero_torque",
            "com.bxi.basic_actions/zero_torque",
        )
    ] is None
