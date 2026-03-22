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

"""Batch-level preprocessor steps that convert joint-space actions to EE-space actions on the fly.

These steps are inserted into the diffusion policy preprocessor pipeline *before* normalization.
They are no-ops at inference time because `select_action` removes the `action` key from the batch
before calling the preprocessor, so the step simply passes the transition through unchanged.

Three modes are supported:
  - ``ee_pose_abs``:         action[t] = FK(teleop_joints[t])  → (x, y, z, wx, wy, wz, gripper_pos)
  - ``ee_pose_delta``:       action[t] = relative SE(3) from follower EE[t] to teleop EE[t]
                             → (delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz, gripper_pos)
  - ``ee_pose_chunk_delta``: action[t] = relative SE(3) from follower EE[0] to teleop EE[t]
                             (i.e. all steps are expressed relative to the initial follower EE state)
                             → (delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz, gripper_pos)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.rotation import Rotation


def _batched_fk(
    kinematics: RobotKinematics,
    joints: np.ndarray,
) -> np.ndarray:
    """Apply FK to a (N, D) array of joint positions (degrees).

    Returns an (N, 4, 4) array of SE(3) transformation matrices.
    """
    n = joints.shape[0]
    transforms = np.empty((n, 4, 4), dtype=np.float64)
    for i in range(n):
        transforms[i] = kinematics.forward_kinematics(joints[i])
    return transforms


def _se3_to_ee_tensor(transforms: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) SE(3) matrices + (N,) gripper values to (N, 7) EE array.

    Layout: [x, y, z, wx, wy, wz, gripper_pos]
    """
    n = transforms.shape[0]
    ee = np.empty((n, 7), dtype=np.float32)
    for i in range(n):
        T = transforms[i]
        ee[i, :3] = T[:3, 3]
        ee[i, 3:6] = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    ee[:, 6] = gripper
    return ee


@dataclass
class JointActionToAbsEEStep(ProcessorStep):
    """Replace joint-space action with absolute EE pose via FK.

    For each action step, computes FK(teleop_joints) and outputs
    ``(x, y, z, wx, wy, wz, gripper_pos)`` — a 7-dimensional EE pose.

    This step is a no-op when ``action`` is not present in the transition
    (i.e. during inference).

    Attributes:
        urdf_path: Path to the robot URDF (serialized to JSON).
        target_frame_name: End-effector frame name in the URDF.
        motor_names: Ordered motor names matching action columns.
        kinematics_leader: Built from the above fields in ``__post_init__``.
    """

    urdf_path: str = ""
    target_frame_name: str = "gripper_frame_link"
    motor_names: list[str] = field(default_factory=list)
    kinematics_leader: RobotKinematics = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.urdf_path:
            self.kinematics_leader = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=self.motor_names or None,
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition  # inference path — no action in batch

        # action: (B, horizon, D) tensor of joint positions in degrees
        device = action.device
        dtype = action.dtype
        B, H, D = action.shape
        joints_np = action.detach().cpu().numpy().reshape(B * H, D)

        # Gripper is the last column; pass through unchanged
        gripper_np = joints_np[:, -1]

        transforms = _batched_fk(self.kinematics_leader, joints_np)  # (B*H, 4, 4)
        ee_np = _se3_to_ee_tensor(transforms, gripper_np)  # (B*H, 7)
        ee_tensor = torch.from_numpy(ee_np).to(device=device, dtype=dtype).reshape(B, H, 7)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = ee_tensor
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class JointActionToDeltaEEStep(ProcessorStep):
    """Replace joint-space action with per-step delta EE pose.

    For each action step at horizon index t, computes:
        T_ref  = FK(follower_joints[t])   — from ``aux.curr_state[:, t, :]``
        T_goal = FK(teleop_joints[t])
        delta_pos    = T_goal[:3, 3] - T_ref[:3, 3]
        delta_rotvec = Rotation(T_ref[:3,:3].T @ T_goal[:3,:3]).as_rotvec()

    ``aux.curr_state`` must be present in the batch — it is loaded by the dataset at the same
    delta indices as the action sequence via ``DiffusionConfig.auxiliary_delta_indices``.

    Output per step: ``(delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz, gripper_pos)``

    This step is a no-op when ``action`` is not present in the transition
    (i.e. during inference).

    Attributes:
        urdf_path: Path to the robot URDF (serialized to JSON).
        target_frame_name: End-effector frame name in the URDF.
        motor_names: Ordered motor names matching action columns.
        kinematics_leader/follower: Built from the above fields in ``__post_init__``.
    """

    urdf_path: str = ""
    target_frame_name: str = "gripper_frame_link"
    motor_names: list[str] = field(default_factory=list)
    kinematics_leader: RobotKinematics = field(default=None, init=False, repr=False)
    kinematics_follower: RobotKinematics = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.urdf_path:
            self.kinematics_leader = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=self.motor_names or None,
            )
            self.kinematics_follower = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=self.motor_names or None,
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition  # inference path — no action in batch

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError(
                "JointActionToDeltaEEStep requires 'observation' in the transition."
            )

        # aux.curr_state: (B, H, state_dim) — follower joints co-recorded at each action timestep
        curr_state = observation.get("aux.curr_state")
        if curr_state is None:
            raise ValueError(
                "JointActionToDeltaEEStep requires 'aux.curr_state' in the batch. "
                "Ensure DiffusionConfig.auxiliary_delta_indices is set (ee_action_space='ee_pose_delta') "
                "and the dataset is built with resolve_feature_aliases."
            )

        device = action.device
        dtype = action.dtype
        B, H, D = action.shape

        # Per-step follower joints: (B*H, state_dim)
        follower_joints_np = curr_state.detach().cpu().numpy().reshape(B * H, -1)

        # Per-step reference EE poses: (B*H, 4, 4)
        T_ref = _batched_fk(self.kinematics_follower, follower_joints_np)

        # Goal EE poses: (B*H, 4, 4)
        leader_joints_np = action.detach().cpu().numpy().reshape(B * H, D)
        gripper_np = leader_joints_np[:, -1]  # (B*H,)
        T_goals = _batched_fk(self.kinematics_leader, leader_joints_np)

        # delta_pos and delta_R are now per-step (no fixed-reference expansion needed)
        delta_pos = T_goals[:, :3, 3] - T_ref[:, :3, 3]  # (B*H, 3)
        delta_R = T_ref[:, :3, :3].transpose(0, 2, 1) @ T_goals[:, :3, :3]  # (B*H, 3, 3)
        delta_rotvec = np.stack(
            [Rotation.from_matrix(delta_R[i]).as_rotvec() for i in range(B * H)], axis=0
        )  # (B*H, 3)

        ee_np = np.concatenate([delta_pos, delta_rotvec, gripper_np[:, None]], axis=-1).astype(
            np.float32
        )  # (B*H, 7)
        ee_tensor = torch.from_numpy(ee_np).to(device=device, dtype=dtype).reshape(B, H, 7)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = ee_tensor
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class JointActionToChunkDeltaEEStep(ProcessorStep):
    """Replace joint-space action with per-step delta EE pose relative to the initial follower state.

    For each action step at horizon index t, computes:
        T_ref    = FK(follower_joints[0])   — fixed to the first step of ``aux.curr_state``
        T_goal[t] = FK(teleop_joints[t])   — per-step teleop target
        delta_pos[t]    = T_goal[t][:3, 3] - T_ref[:3, 3]
        delta_rotvec[t] = Rotation(T_ref[:3,:3].T @ T_goal[t][:3,:3]).as_rotvec()

    Unlike ``JointActionToDeltaEEStep`` which uses a per-step follower reference, this step anchors
    all deltas to the follower EE at the *start* of the action chunk (t=0). The model therefore
    learns to predict "how far is each teleop step from where the follower arm was when the chunk
    began?"

    ``aux.curr_state`` must be present in the batch — it is loaded by the dataset at the same
    delta indices as the action sequence via ``DiffusionConfig.auxiliary_delta_indices``.

    Output per step: ``(delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz, gripper_pos)``

    This step is a no-op when ``action`` is not present in the transition
    (i.e. during inference).

    Attributes:
        urdf_path: Path to the robot URDF (serialized to JSON).
        target_frame_name: End-effector frame name in the URDF.
        motor_names: Ordered motor names matching action columns.
        kinematics_leader/follower: Built from the above fields in ``__post_init__``.
    """

    urdf_path: str = ""
    target_frame_name: str = "gripper_frame_link"
    motor_names: list[str] = field(default_factory=list)
    kinematics_leader: RobotKinematics = field(default=None, init=False, repr=False)
    kinematics_follower: RobotKinematics = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.urdf_path:
            self.kinematics_leader = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=self.motor_names or None,
            )
            self.kinematics_follower = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
                joint_names=self.motor_names or None,
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition  # inference path — no action in batch

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError(
                "JointActionToChunkDeltaEEStep requires 'observation' in the transition."
            )

        # aux.curr_state: (B, H, state_dim) — follower joints co-recorded at each action timestep
        curr_state = observation.get("aux.curr_state")
        if curr_state is None:
            raise ValueError(
                "JointActionToChunkDeltaEEStep requires 'aux.curr_state' in the batch. "
                "Ensure DiffusionConfig.auxiliary_delta_indices is set and the dataset is "
                "built with resolve_feature_aliases."
            )

        device = action.device
        dtype = action.dtype
        B, H, D = action.shape

        # Chunk-initial follower joints: take t=0, tile across H → (B*H, state_dim)
        follower_joints_np = curr_state.detach().cpu().numpy()  # (B, H, state_dim)
        follower_joints_t0 = follower_joints_np[:, 0, :]  # (B, state_dim) — first step only
        follower_joints_t0_tiled = np.repeat(follower_joints_t0, H, axis=0)  # (B*H, state_dim)

        # Fixed-per-chunk reference EE poses: (B*H, 4, 4)
        T_ref = _batched_fk(self.kinematics_follower, follower_joints_t0_tiled)

        # Per-step teleop goal joints: (B*H, D)
        leader_joints_np = action.detach().cpu().numpy()  # (B, H, D)
        gripper_np = leader_joints_np[:, :, -1].reshape(B * H)  # (B*H,) per-step gripper
        leader_joints_flat = leader_joints_np.reshape(B * H, D)

        # Per-step goal EE poses: (B*H, 4, 4)
        T_goals = _batched_fk(self.kinematics_leader, leader_joints_flat)

        delta_pos = T_goals[:, :3, 3] - T_ref[:, :3, 3]  # (B*H, 3)
        delta_R = T_ref[:, :3, :3].transpose(0, 2, 1) @ T_goals[:, :3, :3]  # (B*H, 3, 3)
        delta_rotvec = np.stack(
            [Rotation.from_matrix(delta_R[i]).as_rotvec() for i in range(B * H)], axis=0
        )  # (B*H, 3)

        ee_np = np.concatenate([delta_pos, delta_rotvec, gripper_np[:, None]], axis=-1).astype(
            np.float32
        )  # (B*H, 7)
        ee_tensor = torch.from_numpy(ee_np).to(device=device, dtype=dtype).reshape(B, H, 7)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = ee_tensor
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
