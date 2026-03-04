#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, transition_to_robot_action
from lerobot.processor.factory import make_default_robot_action_processor
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.utils.rotation import Rotation

# EE action key order — must match the 7D tensor produced by the diffusion policy
# when trained with ee_pose_abs or ee_pose_delta action space.
EE_ACTION_NAMES = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]


def ee_tensor_obs_to_transition(
    action_observation: tuple[torch.Tensor, dict],
) -> EnvTransition:
    """Convert a (tensor, obs_dict) pair into an EnvTransition with ee.* action keys.

    Use this as the ``to_transition`` argument when building a ``RobotProcessorPipeline``
    for EE action spaces (``ee_pose_abs`` or ``ee_pose_delta``).  It replaces the default
    converter, which requires a pre-built dict and would reject a raw tensor.

    Args:
        action_observation: Tuple of (action_tensor, observation_dict).
            ``action_tensor`` is a 1-D tensor of length 7 with layout
            ``[x, y, z, wx, wy, wz, gripper_pos]``.

    Returns:
        An ``EnvTransition`` with the action stored as a ``ee.*`` dict and the
        observation stored under ``TransitionKey.OBSERVATION``.
    """
    tensor, obs = action_observation
    ee_dict = {name: float(tensor[i]) for i, name in enumerate(EE_ACTION_NAMES)}
    return create_transition(action=ee_dict, observation=obs)


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        _last_pos: Internal state storing the last commanded position.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")

        self._last_pos = pos

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("delta_ee_to_absolute_ee")
@dataclass
class DeltaEEToAbsoluteEEStep(RobotActionProcessorStep):
    """
    Converts a delta end-effector action into an absolute EE target pose.

    Used at inference time when the policy was trained with ``ee_pose_delta`` action space.
    It reads the current follower joint positions from the observation, applies FK to obtain
    the current EE pose (T_ref), then composes it with the policy's delta output to produce
    an absolute EE target for the downstream ``InverseKinematicsEEToJoints`` step.

    The caller (inference script) is responsible for mapping the policy output tensor to a
    dict with ``ee.*`` keys before passing it to this processor.  Example:
        ee_names = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]
        ee_dict = {name: float(action_tensor[i]) for i, name in enumerate(ee_names)}
        robot_action = robot_action_processor((ee_dict, obs))

    Input action keys (delta values):
        ee.x, ee.y, ee.z       — position delta in meters
        ee.wx, ee.wy, ee.wz    — rotation delta as rotation vector (radians)
        ee.gripper_pos         — gripper position, passed through unchanged

    Output action keys (absolute values, same names overwritten):
        ee.x, ee.y, ee.z       — absolute EE position
        ee.wx, ee.wy, ee.wz    — absolute EE orientation as rotation vector
        ee.gripper_pos         — unchanged

    Attributes:
        kinematics:           FK solver for the follower robot.
        follower_motor_names: Motor names in the same order as observation joint positions.
    """

    kinematics: RobotKinematics
    follower_motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if obs is None:
            raise ValueError("DeltaEEToAbsoluteEEStep requires an observation in the transition.")

        # Build current joint array (arm joints only, skip gripper for FK)
        current_joints = np.array(
            [float(obs[f"{name}.pos"]) for name in self.follower_motor_names if name != "gripper"],
            dtype=float,
        )

        # FK: current follower EE pose
        T_ref = self.kinematics.forward_kinematics(current_joints)

        # Read delta EE from action
        delta_pos = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        delta_R = Rotation.from_rotvec([action["ee.wx"], action["ee.wy"], action["ee.wz"]]).as_matrix()

        # Compose: T_goal = T_ref ⊕ delta_T  (delta expressed in world frame)
        T_goal = np.eye(4, dtype=float)
        T_goal[:3, :3] = T_ref[:3, :3] @ delta_R
        T_goal[:3, 3] = T_ref[:3, 3] + delta_pos

        abs_pos = T_goal[:3, 3]
        abs_rotvec = Rotation.from_matrix(T_goal[:3, :3]).as_rotvec()

        action["ee.x"] = float(abs_pos[0])
        action["ee.y"] = float(abs_pos[1])
        action["ee.z"] = float(abs_pos[2])
        action["ee.wx"] = float(abs_rotvec[0])
        action["ee.wy"] = float(abs_rotvec[1])
        action["ee.wz"] = float(abs_rotvec[2])
        # ee.gripper_pos passes through unchanged
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Key names are unchanged (ee.* stays ee.*); only semantics change delta → absolute.
        return features


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        # TODO: This assumes gripper is the last specified joint in the robot
        gripper_pos = float(np.clip(q_raw[-1] + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


def make_policy_robot_action_processor(
    ee_action_space: str,
    motor_names: list[str],
    ee_urdf_path: str | None = None,
    end_effector_bounds: dict | None = None,
) -> RobotProcessorPipeline:
    """Build a robot_action_processor pipeline that converts policy output to joint commands.

    Args:
        ee_action_space: One of "joint_pos", "ee_pose_abs", "ee_pose_delta".
        motor_names: Ordered list of motor names (including gripper).
        ee_urdf_path: Path to the robot URDF.  Required when ee_action_space != "joint_pos".
        end_effector_bounds: Optional dict with "min"/"max" numpy arrays (3D position bounds)
            passed to EEBoundsAndSafety.  Defaults to conservative workspace bounds if None.

    Returns:
        A RobotProcessorPipeline whose input type matches the action space:
          - joint_pos: (RobotAction dict, RobotObservation) → joint dict  (identity)
          - ee_pose_abs / ee_pose_delta: (action tensor, RobotObservation) → joint dict
    """
    if ee_action_space == "joint_pos":
        return make_default_robot_action_processor()

    if ee_urdf_path is None:
        raise ValueError("ee_urdf_path must be set when ee_action_space != 'joint_pos'")

    arm_names = [n for n in motor_names if n != "gripper"]
    kinematics = RobotKinematics(urdf_path=ee_urdf_path, joint_names=arm_names)

    if end_effector_bounds is None:
        end_effector_bounds = {
            "min": np.array([-0.5, -0.5, 0.0]),
            "max": np.array([0.5, 0.5, 0.8]),
        }

    steps = []
    if ee_action_space == "ee_pose_delta":
        steps.append(DeltaEEToAbsoluteEEStep(
            kinematics=kinematics,
            follower_motor_names=motor_names,
        ))
    elif ee_action_space != "ee_pose_abs":
        raise ValueError(
            f"Unknown ee_action_space '{ee_action_space}'. "
            "Choose from 'joint_pos', 'ee_pose_abs', 'ee_pose_delta'."
        )

    steps.extend([
        EEBoundsAndSafety(end_effector_bounds=end_effector_bounds),
        InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motor_names),
    ])

    return RobotProcessorPipeline(
        steps=steps,
        to_transition=ee_tensor_obs_to_transition,
        to_output=transition_to_robot_action,
    )


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None
