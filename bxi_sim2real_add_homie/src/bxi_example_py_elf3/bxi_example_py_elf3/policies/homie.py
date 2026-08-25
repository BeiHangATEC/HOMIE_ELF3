"""HOMIE crouch-and-walk policy for the ELF3 controller."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.framework.inference import (
    HistoryBuffer,
    InferenceFrame,
    InferenceRuntime,
    JointPolicy,
    ModelSpec,
    PolicyJointContract,
    PolicyOutput,
    default_runtime,
)
from bxi_example_py_elf3.framework.joints import (
    JointLayout,
    JointParameterSet,
)

from .joints import ELF3_POLICY_JOINTS


HOMIE_ACTION_JOINT_NAMES = (
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
)

HOMIE_OBSERVATION_JOINT_NAMES = HOMIE_ACTION_JOINT_NAMES + (
    "waist_y_joint",
    "waist_z_joint",
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)

HOMIE_ACTION_JOINTS = JointLayout(
    HOMIE_ACTION_JOINT_NAMES,
    label="HOMIE 12-joint action",
)
HOMIE_OBSERVATION_JOINTS = JointLayout(
    HOMIE_OBSERVATION_JOINT_NAMES,
    label="HOMIE 28-joint observation",
)


# Rows are (joint name, default position, kp, kd, action scale).  Keeping the
# values next to names makes the 28-to-29-joint boundary directly reviewable.
HOMIE_PARAMETERS = JointParameterSet.from_rows(
    ELF3_POLICY_JOINTS,
    (
        ("waist_y_joint", 0.0, 300.0, 5.0, 0.0),
        ("waist_x_joint", 0.0, 300.0, 5.0, 0.0),
        ("waist_z_joint", 0.0, 300.0, 5.0, 0.0),
        ("l_hip_y_joint", -0.1, 100.0, 2.0, 0.25),
        ("l_hip_x_joint", 0.0, 100.0, 2.0, 0.25),
        ("l_hip_z_joint", 0.0, 100.0, 2.0, 0.25),
        ("l_knee_y_joint", 0.3, 150.0, 4.0, 0.25),
        ("l_ankle_y_joint", -0.2, 40.0, 2.0, 0.25),
        ("l_ankle_x_joint", 0.0, 40.0, 2.0, 0.25),
        ("r_hip_y_joint", -0.1, 100.0, 2.0, 0.25),
        ("r_hip_x_joint", 0.0, 100.0, 2.0, 0.25),
        ("r_hip_z_joint", 0.0, 100.0, 2.0, 0.25),
        ("r_knee_y_joint", 0.3, 150.0, 4.0, 0.25),
        ("r_ankle_y_joint", -0.2, 40.0, 2.0, 0.25),
        ("r_ankle_x_joint", 0.0, 40.0, 2.0, 0.25),
        ("l_shoulder_y_joint", 0.0, 200.0, 4.0, 0.0),
        ("l_shoulder_x_joint", 0.0, 200.0, 4.0, 0.0),
        ("l_shoulder_z_joint", 0.0, 200.0, 4.0, 0.0),
        ("l_elbow_y_joint", 0.0, 100.0, 1.0, 0.0),
        ("l_wrist_x_joint", 0.0, 20.0, 0.5, 0.0),
        ("l_wrist_y_joint", 0.0, 20.0, 0.5, 0.0),
        ("l_wrist_z_joint", 0.0, 20.0, 0.5, 0.0),
        ("r_shoulder_y_joint", 0.0, 200.0, 4.0, 0.0),
        ("r_shoulder_x_joint", 0.0, 200.0, 4.0, 0.0),
        ("r_shoulder_z_joint", 0.0, 200.0, 4.0, 0.0),
        ("r_elbow_y_joint", 0.0, 100.0, 1.0, 0.0),
        ("r_wrist_x_joint", 0.0, 20.0, 0.5, 0.0),
        ("r_wrist_y_joint", 0.0, 20.0, 0.5, 0.0),
        ("r_wrist_z_joint", 0.0, 20.0, 0.5, 0.0),
    ),
)


HOMIE_ONNX_METADATA = {
    "policy_family": "homie_him_elf3",
    "source_checkpoint_sha256": (
        "085ce1758c2e65d7f0914ab2fcc6de16ba71eeda433f8eba550418638dcd6579"
    ),
    "observation_joint_names": ",".join(HOMIE_OBSERVATION_JOINT_NAMES),
    "action_joint_names": ",".join(HOMIE_ACTION_JOINT_NAMES),
    "history_length": "6",
    "one_step_observation_size": "78",
    "action_scale": "0.25",
    "height_min_m": "0.40",
    "height_max_m": "1.01",
}


class PolicySafetyError(RuntimeError):
    """A HOMIE runtime value violated the sim2real safety contract."""


class HomiePolicy(JointPolicy):
    """Six-frame HIM policy with a full named ELF3 motor target."""

    INPUT_NAME = "obs"
    OUTPUT_NAME = "actions"
    HISTORY_LENGTH = 6
    ONE_STEP_OBSERVATION_SIZE = 78
    INPUT_SIZE = HISTORY_LENGTH * ONE_STEP_OBSERVATION_SIZE
    ACTION_SIZE = len(HOMIE_ACTION_JOINT_NAMES)
    HEIGHT_MIN_M = 0.40
    HEIGHT_MAX_M = 1.01
    COMMAND_MIN = -0.5
    COMMAND_MAX = 0.5
    ACTION_CLIP = 100.0

    joint_contract = PolicyJointContract(
        observation=HOMIE_OBSERVATION_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "onnxruntime",
    ) -> None:
        super().__init__()
        self._runtime = runtime or default_runtime()
        spec = (
            model
            if isinstance(model, ModelSpec)
            else ModelSpec.onnx(
                model,
                input_names=(self.INPUT_NAME,),
                output_names=(self.OUTPUT_NAME,),
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)
        try:
            self._validate_backend_contract()
        except Exception:
            self._backend.close()
            raise

        self._parameters = HOMIE_PARAMETERS
        self._input = np.zeros((1, self.INPUT_SIZE), dtype=np.float32)
        self._inputs = {self.INPUT_NAME: self._input}
        self._one_step = np.zeros(
            self.ONE_STEP_OBSERVATION_SIZE,
            dtype=np.float32,
        )
        self._history = HistoryBuffer(
            self.HISTORY_LENGTH,
            self.ONE_STEP_OBSERVATION_SIZE,
            dtype=np.float32,
        )
        self._previous_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self._candidate_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self._gravity = np.zeros(3, dtype=np.float32)
        self._target = self._target_buffer.position
        self._observation_default = np.asarray(
            [
                self._parameters.default_position[ELF3_POLICY_JOINTS.index(name)]
                for name in HOMIE_OBSERVATION_JOINT_NAMES
            ],
            dtype=np.float32,
        )
        self._action_indices = np.asarray(
            [ELF3_POLICY_JOINTS.index(name) for name in HOMIE_ACTION_JOINT_NAMES],
            dtype=np.intp,
        )
        self._waist_x_index = ELF3_POLICY_JOINTS.index("waist_x_joint")
        self._height_command = self.HEIGHT_MAX_M
        np.copyto(self._target, self._parameters.default_position)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    @property
    def inputs(self) -> Mapping[str, NDArray[np.generic]]:
        return self._inputs

    @property
    def parameters(self) -> JointParameterSet:
        return self._parameters

    @property
    def height_command(self) -> float:
        return self._height_command

    def set_height_command(self, height_m: float) -> None:
        self._height_command = self._validated_height(height_m)

    def _validate_backend_contract(self) -> None:
        if tuple(self._backend.input_names) != (self.INPUT_NAME,):
            raise ValueError(
                "HOMIE backend inputs must be exactly ('obs',)"
            )
        if tuple(self._backend.output_names) != (self.OUTPUT_NAME,):
            raise ValueError(
                "HOMIE backend outputs must be exactly ('actions',)"
            )
        self._validate_batched_shape(
            "input",
            tuple(self._backend.input_shape(self.INPUT_NAME)),
            self.INPUT_SIZE,
        )
        self._validate_batched_shape(
            "output",
            tuple(self._backend.output_shape(self.OUTPUT_NAME)),
            self.ACTION_SIZE,
        )

        metadata = dict(self._backend.metadata)
        missing = tuple(key for key in HOMIE_ONNX_METADATA if key not in metadata)
        if missing:
            raise ValueError(
                "HOMIE ONNX metadata is missing required keys: "
                + ", ".join(missing)
            )
        mismatches = tuple(
            key
            for key, expected in HOMIE_ONNX_METADATA.items()
            if metadata[key] != expected
        )
        if mismatches:
            details = "; ".join(
                f"{key}: expected={HOMIE_ONNX_METADATA[key]!r}, "
                f"got={metadata[key]!r}"
                for key in mismatches
            )
            raise ValueError(f"HOMIE ONNX metadata mismatch: {details}")

    @staticmethod
    def _validate_batched_shape(
        label: str,
        actual: tuple[object, ...],
        width: int,
    ) -> None:
        if len(actual) != 2 or actual[1] != width:
            raise ValueError(
                f"HOMIE backend {label} shape must be (batch, {width}), "
                f"got {actual}"
            )
        batch = actual[0]
        if isinstance(batch, (int, np.integer)) and not isinstance(
            batch,
            (bool, np.bool_),
        ):
            if int(batch) != 1:
                raise ValueError(
                    f"HOMIE backend {label} batch must be dynamic or one, "
                    f"got {batch!r}"
                )
        elif batch is not None and not isinstance(batch, str):
            raise ValueError(
                f"HOMIE backend {label} batch must be dynamic or one, "
                f"got {batch!r}"
            )

    @staticmethod
    def _finite_vector(
        value: object,
        shape: tuple[int, ...],
        label: str,
    ) -> NDArray[np.floating]:
        array = np.asarray(value)
        if array.shape != shape:
            raise PolicySafetyError(
                f"HOMIE {label} shape is {array.shape}, expected {shape}"
            )
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype,
            np.complexfloating,
        ):
            raise PolicySafetyError(f"HOMIE {label} must be real numeric values")
        if not np.isfinite(array).all():
            raise PolicySafetyError(f"HOMIE {label} must be finite")
        return array

    @classmethod
    def _validated_height(cls, height_m: object) -> float:
        array = np.asarray(height_m)
        if (
            array.shape != ()
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
        ):
            raise PolicySafetyError("HOMIE height command must be a real scalar")
        height = float(array)
        if not np.isfinite(height):
            raise PolicySafetyError("HOMIE height command must be finite")
        if not cls.HEIGHT_MIN_M <= height <= cls.HEIGHT_MAX_M:
            raise PolicySafetyError(
                "HOMIE height command must stay inside the training range "
                f"[{cls.HEIGHT_MIN_M:.2f}, {cls.HEIGHT_MAX_M:.2f}] m"
            )
        return height

    def _project_gravity(self, quaternion: NDArray[np.floating]) -> None:
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1.0e-6:
            raise PolicySafetyError("HOMIE body quaternion has invalid norm")
        w, x, y, z = np.asarray(quaternion, dtype=np.float64) / norm
        self._gravity[0] = 2.0 * (w * y - x * z)
        self._gravity[1] = -2.0 * (w * x + y * z)
        self._gravity[2] = 2.0 * (x * x + y * y) - 1.0

    def _build_one_step(self, frame: InferenceFrame, joints, height_m: float) -> None:
        if frame.command is None:
            raise PolicySafetyError("HOMIE requires a velocity command")
        command = self._finite_vector(frame.command, (3,), "command")
        if np.any(command < self.COMMAND_MIN) or np.any(
            command > self.COMMAND_MAX
        ):
            raise PolicySafetyError(
                "HOMIE velocity command is outside the training range "
                "[-0.5, 0.5]"
            )
        angular_velocity = self._finite_vector(
            frame.angular_velocity,
            (3,),
            "angular velocity",
        )
        quaternion = self._finite_vector(
            frame.quat_wxyz,
            (4,),
            "body quaternion",
        )
        if not np.isfinite(joints.position).all() or not np.isfinite(
            joints.velocity
        ).all():
            raise PolicySafetyError("HOMIE joint observation must be finite")

        self._project_gravity(quaternion)
        one_step = self._one_step
        one_step[0:3] = command * np.asarray((2.0, 2.0, 0.5), dtype=np.float32)
        one_step[3] = height_m
        one_step[4:7] = angular_velocity * 0.5
        one_step[7:10] = self._gravity
        np.subtract(
            joints.position,
            self._observation_default,
            out=one_step[10:38],
        )
        np.multiply(joints.velocity, 0.05, out=one_step[38:66])
        one_step[66:78] = self._previous_action
        if not np.isfinite(one_step).all():
            raise PolicySafetyError("HOMIE observation became non-finite")

    def reset(self, frame: InferenceFrame) -> None:
        joints = self.bind_joints(frame)
        self._previous_action.fill(0.0)
        self._candidate_action.fill(0.0)
        self._history.clear()
        self._build_one_step(frame, joints, self._height_command)
        # Training starts each episode with an empty history and appends the
        # current observation once: five zero frames followed by this frame.
        self._history.preview_append_into(self._one_step, self._input[0])
        np.copyto(self._target, self._parameters.default_position)
        self._target[self._waist_x_index] = 0.0
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    def _decode_action(
        self,
        outputs: Mapping[str, NDArray[np.generic]],
    ) -> NDArray[np.float32]:
        if set(outputs) != {self.OUTPUT_NAME}:
            raise PolicySafetyError(
                "HOMIE backend outputs do not match the actions contract"
            )
        raw = np.asarray(outputs[self.OUTPUT_NAME])
        expected = (1, self.ACTION_SIZE)
        if raw.shape != expected:
            raise PolicySafetyError(
                f"HOMIE action tensor shape is {raw.shape}, expected {expected}"
            )
        if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
            raise PolicySafetyError("HOMIE action tensor must contain finite floats")
        np.clip(
            raw[0],
            -self.ACTION_CLIP,
            self.ACTION_CLIP,
            out=self._candidate_action,
        )
        return self._candidate_action

    def _publish_action(self, action: NDArray[np.float32]) -> PolicyOutput:
        np.copyto(self._target, self._parameters.default_position)
        self._target[self._action_indices] += action * 0.25
        self._target[self._waist_x_index] = 0.0
        if not np.isfinite(self._target).all():
            raise PolicySafetyError("HOMIE joint target became non-finite")
        return self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        height_command: float | None = None,
        advance: bool = True,
    ) -> PolicyOutput:
        if not np.isfinite(dt) or dt < 0.0:
            raise PolicySafetyError("HOMIE policy dt must be finite and non-negative")
        height = self._validated_height(
            self._height_command if height_command is None else height_command
        )
        joints = self.bind_joints(frame)
        self._build_one_step(frame, joints, height)
        self._history.preview_append_into(self._one_step, self._input[0])
        try:
            outputs = self._backend.run(self._inputs)
        except Exception as exc:
            raise PolicySafetyError(f"HOMIE inference failed: {exc}") from exc
        action = self._decode_action(outputs)
        output = self._publish_action(action)
        if advance:
            self._history.append(self._one_step)
            np.copyto(self._previous_action, action)
            self._height_command = height
        return output

    def decode_into(self, outputs: Mapping[str, NDArray[np.generic]]) -> None:
        action = self._decode_action(outputs)
        self._publish_action(action)
        np.copyto(self._previous_action, action)

    def close(self) -> None:
        self._backend.close()


__all__ = [
    "HOMIE_ACTION_JOINT_NAMES",
    "HOMIE_ACTION_JOINTS",
    "HOMIE_OBSERVATION_JOINT_NAMES",
    "HOMIE_OBSERVATION_JOINTS",
    "HOMIE_ONNX_METADATA",
    "HOMIE_PARAMETERS",
    "HomiePolicy",
    "PolicySafetyError",
]
