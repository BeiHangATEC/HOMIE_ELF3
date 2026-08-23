import importlib.util
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGGED_GYM_ROOT = REPO_ROOT / "HomieRL/legged_gym"
CONTRACT_PATH = LEGGED_GYM_ROOT / "legged_gym/utils/sim2sim_contract.py"
RUNNER_PATH = LEGGED_GYM_ROOT / "legged_gym/scripts/sim2sim_elf3.py"
URDF_PATH = LEGGED_GYM_ROOT / "resources/robots/elf3_description/urdf/elf3.urdf"
MJCF_PATH = LEGGED_GYM_ROOT / "resources/robots/elf3_description/mjcf/elf3_fixed_waist.xml"
MANIFEST_PATH = MJCF_PATH.with_name("derived_asset_manifest.json")
sys.path.insert(0, str(LEGGED_GYM_ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY_JOINTS = [
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
]
UPPER_JOINTS = [
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]


def fake_config(task):
    defaults = {name: 0.0 for name in POLICY_JOINTS + UPPER_JOINTS}
    for side in ("l", "r"):
        defaults[f"{side}_hip_y_joint"] = -0.1
        defaults[f"{side}_knee_y_joint"] = 0.3
        defaults[f"{side}_ankle_y_joint"] = -0.2
    velocity_ranges = {
        "elf3": ([-0.8, 1.2], [-0.5, 0.5], [-0.8, 0.8]),
        "elf3_height": ([-0.8, 1.2], [-0.5, 0.5], [-0.8, 0.8]),
        "elf3_walk": ([-0.5, 0.5], [-0.3, 0.3], [-0.5, 0.5]),
        "elf3_walk_toeout": ([-0.5, 0.5], [-0.3, 0.3], [-0.5, 0.5]),
    }
    velocity_x, velocity_y, yaw_rate = velocity_ranges[task]
    commands = SimpleNamespace(
        height_target=1.01 if task == "elf3" else 1.0,
        height_slew_rate=0.2,
        ranges=SimpleNamespace(
            lin_vel_x=velocity_x,
            lin_vel_y=velocity_y,
            ang_vel_yaw=yaw_rate,
            height=[-0.71, 0.0] if task == "elf3" else [0.0, 0.0],
        ),
    )
    if task in ("elf3_height", "elf3_walk", "elf3_walk_toeout"):
        commands.height_min = 0.3
        commands.height_max = 1.0
    if task == "elf3_walk_toeout":
        commands.toe_out_start_height = 0.735
        commands.toe_out_full_angle_height = 0.50
        commands.toe_out_max_angle_deg = 15.0
    return SimpleNamespace(
        asset=SimpleNamespace(
            policy_dof_names=POLICY_JOINTS,
            upper_body_dof_names=UPPER_JOINTS,
        ),
        init_state=SimpleNamespace(default_joint_angles=defaults),
        control=SimpleNamespace(
            stiffness={
                "hip_z": 100, "hip_x": 100, "hip_y": 100, "knee": 150,
                "ankle": 40, "shoulder": 200, "wrist": 20, "elbow": 100,
            },
            damping={
                "hip_z": 2, "hip_x": 2, "hip_y": 2, "knee": 4,
                "ankle": 2, "shoulder": 4, "wrist": 0.5, "elbow": 1,
            },
            action_scale=0.25,
            decimation=4,
        ),
        commands=commands,
        rewards=SimpleNamespace(
            least_feet_distance_lateral=0.20,
            most_feet_distance_lateral=0.35,
        ),
        env=SimpleNamespace(
            num_one_step_observations=74,
            num_actor_history=6,
            num_observations=444,
            num_actions=12,
        ),
        normalization=SimpleNamespace(
            obs_scales=SimpleNamespace(lin_vel=2.0, ang_vel=0.5, dof_pos=1.0, dof_vel=0.05),
            clip_observations=100.0,
            clip_actions=100.0,
        ),
        sim=SimpleNamespace(dt=0.005),
    )


class Elf3Sim2SimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_module = load_module("elf3_sim2sim_contract", CONTRACT_PATH)
        cls.runner = load_module("elf3_sim2sim_runner", RUNNER_PATH)

    def build_contract(self, task, policy_path):
        return self.contract_module.build_contract(
            task,
            fake_config(task),
            policy_path,
            MJCF_PATH,
            URDF_PATH,
        )

    def test_derived_model_matches_fixed_waist_contract(self):
        urdf = ET.parse(URDF_PATH).getroot()
        urdf_types = {joint.get("name"): joint.get("type") for joint in urdf.findall("joint")}
        for name in ("waist_y_joint", "waist_x_joint", "waist_z_joint", "head_z_joint", "head_y_joint"):
            self.assertEqual(urdf_types[name], "fixed")
        self.assertEqual(sum(value == "revolute" for value in urdf_types.values()), 26)

        mjcf = ET.parse(MJCF_PATH).getroot()
        mjcf_joints = {
            joint.get("name") for joint in mjcf.findall(".//joint") if joint.get("name")
        }
        actuators = [motor.get("joint") for motor in mjcf.findall("./actuator/motor")]
        self.assertEqual(mjcf_joints, set(POLICY_JOINTS + UPPER_JOINTS))
        self.assertEqual(actuators, POLICY_JOINTS + UPPER_JOINTS)
        self.assertEqual(mjcf.find("./asset/mesh[@name='l_hand']").get("file"), "../urdf/meshes/l_hand.STL")
        self.assertEqual(mjcf.find("./asset/mesh[@name='r_hand']").get("file"), "../urdf/meshes/r_hand.STL")

        model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.assertEqual((model.nq, model.nv, model.nu), (33, 32, 26))
        self.assertAlmostEqual(model.opt.timestep, 0.005)
        self.assertAlmostEqual(float(model.body_mass.sum()), 43.222480280000006)

        urdf_links = {link.get("name"): link for link in urdf.findall("link")}
        for body in mjcf.findall(".//body"):
            name = body.get("name")
            urdf_inertial = urdf_links[name].find("inertial")
            mjcf_inertial = body.find("inertial")
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertAlmostEqual(
                float(mjcf_inertial.get("mass")),
                float(urdf_inertial.find("mass").get("value")),
                delta=1.0e-5,
            )
            expected_com = np.fromstring(urdf_inertial.find("origin").get("xyz"), sep=" ")
            np.testing.assert_allclose(model.body_ipos[body_id], expected_com, atol=1.0e-6)
            inertia = urdf_inertial.find("inertia")
            expected_tensor = np.asarray(
                [
                    [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
                    [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
                    [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
                ]
            )
            np.testing.assert_allclose(
                np.sort(model.body_inertia[body_id]),
                np.linalg.eigvalsh(expected_tensor),
                rtol=2.0e-4,
                atol=1.0e-8,
            )

        urdf_joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
        mjcf_joint_nodes = {
            joint.get("name"): joint for joint in mjcf.findall(".//joint") if joint.get("name")
        }
        motors = {motor.get("joint"): motor for motor in mjcf.findall("./actuator/motor")}
        mjcf_bodies = {body.get("name"): body for body in mjcf.findall(".//body")}
        for name in POLICY_JOINTS + UPPER_JOINTS:
            urdf_joint = urdf_joints[name]
            mjcf_joint = mjcf_joint_nodes[name]
            lower = float(urdf_joint.find("limit").get("lower"))
            upper = float(urdf_joint.find("limit").get("upper"))
            effort = float(urdf_joint.find("limit").get("effort"))
            child_link = urdf_joint.find("child").get("link")
            np.testing.assert_allclose(
                np.fromstring(mjcf_joint.get("axis"), sep=" "),
                np.fromstring(urdf_joint.find("axis").get("xyz"), sep=" "),
            )
            np.testing.assert_allclose(
                np.fromstring(mjcf_joint.get("range"), sep=" "),
                [lower, upper],
            )
            np.testing.assert_allclose(
                np.fromstring(motors[name].get("ctrlrange"), sep=" "),
                [-effort, effort],
            )
            np.testing.assert_allclose(
                np.fromstring(mjcf_bodies[child_link].get("pos", "0 0 0"), sep=" "),
                np.fromstring(urdf_joint.find("origin").get("xyz"), sep=" "),
            )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["derived_mjcf_sha256"],
            self.contract_module.sha256_file(MJCF_PATH),
        )

    def test_contracts_capture_each_task_command_semantics(self):
        with tempfile.NamedTemporaryFile() as policy:
            contracts = {
                task: self.build_contract(task, policy.name)
                for task in self.contract_module.SUPPORTED_TASKS
            }
        for task, contract in contracts.items():
            self.runner.validate_contract(contract)
            self.assertEqual(contract["task"], task)
            self.assertEqual(contract["policy"]["input_dim"], 444)
            self.assertEqual(contract["policy"]["output_dim"], 12)
            self.assertEqual(contract["observation"]["joint_order"], POLICY_JOINTS + UPPER_JOINTS)
            self.assertEqual(contract["action"]["joint_order"], POLICY_JOINTS)
            self.assertEqual(contract["reset"]["mujoco_root_world_z_m"], 1.05552264)
            self.assertEqual(
                contract["model"]["classification"],
                self.contract_module.DERIVED_MODEL_CLASSIFICATION,
            )
        self.assertEqual(contracts["elf3"]["commands"]["height_m"], [0.3, 1.01])
        self.assertEqual(contracts["elf3_height"]["commands"]["velocity_x_mps"], [0.0, 0.0])
        self.assertEqual(contracts["elf3_height"]["commands"]["height_m"], [0.3, 1.0])
        walk_commands = contracts["elf3_walk"]["commands"]
        self.assertEqual(walk_commands["height_m"], [0.3, 1.0])
        self.assertEqual(walk_commands["default_mode"], "walk")
        self.assertEqual(walk_commands["modes"]["walk"]["height_m"], [1.0, 1.0])
        self.assertEqual(
            walk_commands["modes"]["squat"]["velocity_x_mps"], [0.0, 0.0]
        )
        self.assertEqual(walk_commands["height_slew_rate_mps"], 0.2)
        toeout_contract = contracts["elf3_walk_toeout"]
        self.assertEqual(toeout_contract["commands"], walk_commands)
        self.assertEqual(
            toeout_contract["squat_toe_out_profile"],
            {
                "start_height_m": 0.735,
                "full_angle_height_m": 0.50,
                "left_max_angle_deg": 15.0,
                "right_max_angle_deg": -15.0,
                "feet_distance_range_m": [0.20, 0.35],
            },
        )
        self.assertNotIn("squat_toe_out_profile", contracts["elf3_walk"])

    def test_command_validation_rejects_out_of_distribution_values(self):
        with tempfile.NamedTemporaryFile() as policy:
            height_contract = self.build_contract("elf3_height", policy.name)
            walk_contract = self.build_contract("elf3_walk", policy.name)
        self.contract_module.validate_command(height_contract, 0.0, 0.0, 0.0, 0.3)
        self.contract_module.validate_command(walk_contract, 0.5, -0.3, 0.5, 1.0)
        self.contract_module.validate_command(walk_contract, 0.0, 0.0, 0.0, 0.3)
        with self.assertRaises(ValueError):
            self.contract_module.validate_command(height_contract, 0.1, 0.0, 0.0, 0.8)
        with self.assertRaises(ValueError):
            self.contract_module.validate_command(walk_contract, 0.1, 0.0, 0.0, 0.9)
        with self.assertRaises(ValueError):
            self.contract_module.validate_command(
                walk_contract, 0.0, 0.0, 0.0, 0.9, mode="walk"
            )
        with self.assertRaises(ValueError):
            self.contract_module.validate_command(
                walk_contract, 0.1, 0.0, 0.0, 1.0, mode="squat"
            )
        with self.assertRaises(ValueError):
            self.contract_module.validate_command(walk_contract, float("nan"), 0.0, 0.0, 1.0)

    def test_slider_ranges_and_snapshots_follow_policy_contract(self):
        with tempfile.NamedTemporaryFile() as policy:
            height_contract = self.build_contract("elf3_height", policy.name)
            walk_contract = self.build_contract("elf3_walk", policy.name)

        self.assertEqual(
            self.runner.slider_limits_from_contract(height_contract),
            {
                "x_vel": (0.0, 0.0),
                "y_vel": (0.0, 0.0),
                "yaw_vel": (0.0, 0.0),
                "height": (0.3, 1.0),
            },
        )
        self.assertEqual(
            self.runner.slider_limits_from_contract(walk_contract),
            {
                "x_vel": (-0.5, 0.5),
                "y_vel": (-0.3, 0.3),
                "yaw_vel": (-0.5, 0.5),
                "height": (0.3, 1.0),
            },
        )
        self.assertEqual(
            self.runner.slider_mode_limits_from_contract(walk_contract)["squat"],
            {
                "x_vel": (0.0, 0.0),
                "y_vel": (0.0, 0.0),
                "yaw_vel": (0.0, 0.0),
                "height": (0.3, 1.0),
            },
        )
        command, height = self.runner.command_from_snapshot(
            walk_contract,
            {"x_vel": 0.2, "y_vel": -0.1, "yaw_vel": 0.3, "height": 1.0},
            mode="walk",
        )
        np.testing.assert_allclose(command, [0.2, -0.1, 0.3])
        self.assertEqual(height, 1.0)
        with self.assertRaises(ValueError):
            self.runner.command_from_snapshot(
                height_contract,
                {"x_vel": 0.1, "y_vel": 0.0, "yaw_vel": 0.0, "height": 0.8},
            )
        self.assertAlmostEqual(self.runner.slew_value(1.0, 0.3, 0.2, 0.02), 0.996)
        self.assertAlmostEqual(self.runner.slew_value(0.3, 1.0, 0.2, 0.02), 0.304)

    def test_model_mapping_and_single_frame_are_name_based(self):
        with tempfile.NamedTemporaryFile() as policy:
            contract = self.build_contract("elf3_walk", policy.name)
        model, _ = self.runner.load_model(MJCF_PATH, contract)
        mapping = self.runner.make_mapping(model, contract)
        data = mujoco.MjData(model)
        root_id = self.runner.named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        root_qpos = model.jnt_qposadr[root_id]
        default = np.asarray(contract["reset"]["default_joint_position"])
        data.qpos[root_qpos : root_qpos + 7] = [0.0, 0.0, 1.05552264, 1.0, 0.0, 0.0, 0.0]
        data.qpos[mapping["qpos"]] = default
        mujoco.mj_forward(model, data)
        imu_id = self.runner.named_id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        frame = self.runner.make_frame(
            model,
            data,
            mapping,
            imu_id,
            default,
            np.zeros(12, dtype=np.float32),
            np.asarray([0.2, -0.1, 0.3], dtype=np.float32),
            1.0,
            contract,
        )
        self.assertEqual(frame.shape, (74,))
        np.testing.assert_allclose(frame[:4], [0.4, -0.2, 0.15, 1.0])
        np.testing.assert_allclose(frame[10:36], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(frame[-12:], 0.0)

    def test_toe_out_profile_and_foot_pose_metrics_use_torso_frame(self):
        with tempfile.NamedTemporaryFile() as policy:
            contract = self.build_contract("elf3_walk_toeout", policy.name)
        model, _ = self.runner.load_model(MJCF_PATH, contract)
        mapping = self.runner.make_mapping(model, contract)
        data = mujoco.MjData(model)
        root_id = self.runner.named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        root_qpos = model.jnt_qposadr[root_id]
        data.qpos[root_qpos : root_qpos + 7] = [
            0.0, 0.0, 1.05552264, 1.0, 0.0, 0.0, 0.0
        ]
        data.qpos[mapping["qpos"]] = contract["reset"]["default_joint_position"]
        joint_order = contract["observation"]["joint_order"]
        data.qpos[mapping["qpos"][joint_order.index("l_hip_z_joint")]] = np.radians(15.0)
        data.qpos[mapping["qpos"][joint_order.index("r_hip_z_joint")]] = np.radians(-15.0)
        mujoco.mj_forward(model, data)

        torso_id = self.runner.named_id(
            model, mujoco.mjtObj.mjOBJ_BODY, contract["bodies"]["torso_body"]
        )
        foot_ids = [
            self.runner.named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in contract["bodies"]["foot_bodies"]
        ]
        yaws, distance = self.runner.foot_pose_metrics(
            model, data, torso_id, foot_ids
        )
        np.testing.assert_allclose(np.degrees(yaws), [15.0, -15.0], atol=0.1)
        self.assertGreater(distance, 0.0)

        profile = contract["squat_toe_out_profile"]
        expected = {
            0.735: 0.0,
            0.65: 5.425532,
            0.60: 8.617021,
            0.50: 15.0,
            0.30: 15.0,
        }
        for height, left_degrees in expected.items():
            targets = np.degrees(
                self.runner.squat_toe_out_targets(height, profile)
            )
            np.testing.assert_allclose(
                targets, [left_degrees, -left_degrees], atol=1.0e-5
            )

    def test_invalid_toe_out_profile_is_rejected_without_breaking_old_contracts(self):
        with tempfile.NamedTemporaryFile() as policy:
            old_contract = self.build_contract("elf3_walk", policy.name)
            toeout_contract = self.build_contract("elf3_walk_toeout", policy.name)
        self.runner.validate_contract(old_contract)
        toeout_contract["squat_toe_out_profile"]["right_max_angle_deg"] = 15.0
        with self.assertRaises(RuntimeError):
            self.runner.validate_contract(toeout_contract)

    def test_model_and_policy_hash_mismatches_are_rejected(self):
        with tempfile.NamedTemporaryFile() as policy:
            contract = self.build_contract("elf3", policy.name)
            contract["model"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                self.runner.load_model(MJCF_PATH, contract)
            contract["policy"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                self.runner.load_policy(policy.name, contract)


if __name__ == "__main__":
    unittest.main()
