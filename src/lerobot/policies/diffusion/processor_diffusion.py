#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.joint_to_ee_processor import (
    JointActionToAbsEEStep,
    JointActionToDeltaEEStep,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# Hardcoded motor names per robot type (order must match the action tensor column order).
_MOTOR_NAMES: dict[str, list[str]] = {
    "so101": [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ],
}


def _get_motor_names(robot_type: str) -> list[str]:
    if robot_type not in _MOTOR_NAMES:
        raise NotImplementedError(
            f"EE action space only supports robot types {list(_MOTOR_NAMES)}, got '{robot_type}'. "
            "Add your robot's motor names to _MOTOR_NAMES in processor_diffusion.py."
        )
    return _MOTOR_NAMES[robot_type]


def _build_ee_action_stats(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build a 7D action stats dict by combining 6D EE bounds + gripper from dataset stats."""
    ee_cfg = config.ee_action_stats
    if ee_cfg is None:
        raise ValueError(
            "ee_action_stats must be set when ee_action_space != 'joint_pos'. "
            "Provide 6D bounds (no gripper) as {'action': {'min': [...], 'max': [...]}}."
        )

    ee_min = torch.as_tensor(ee_cfg["action"]["min"], dtype=torch.float32)  # (6,)
    ee_max = torch.as_tensor(ee_cfg["action"]["max"], dtype=torch.float32)  # (6,)

    # Gripper stats from original dataset (last dim of the original joint action)
    if dataset_stats is not None and "action" in dataset_stats:
        gripper_min = torch.as_tensor(dataset_stats["action"]["min"][-1:], dtype=torch.float32)
        gripper_max = torch.as_tensor(dataset_stats["action"]["max"][-1:], dtype=torch.float32)
    else:
        gripper_min = torch.tensor([0.0])
        gripper_max = torch.tensor([100.0])

    combined_stats: dict[str, dict[str, torch.Tensor]] = dict(dataset_stats) if dataset_stats else {}
    combined_stats["action"] = {
        "min": torch.cat([ee_min, gripper_min]),   # (7,)
        "max": torch.cat([ee_max, gripper_max]),   # (7,)
    }
    return combined_stats


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Optionally transforming joint-space actions to EE-space (ee_pose_abs / ee_pose_delta).
    2. Renaming features.
    3. Normalizing the input and output features based on dataset statistics.
    4. Adding a batch dimension.
    5. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Resolve effective normalization stats (may be overridden for EE action spaces).
    if config.ee_action_space != "joint_pos":
        effective_stats = _build_ee_action_stats(config, dataset_stats)
    else:
        effective_stats = dataset_stats

    input_steps: list = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    if config.ee_action_space != "joint_pos":
        if config.ee_urdf_path is None:
            raise ValueError("ee_urdf_path must be set when ee_action_space != 'joint_pos'.")
        motor_names = _get_motor_names(config.ee_robot_type)

        if config.ee_action_space == "ee_pose_abs":
            input_steps.append(
                JointActionToAbsEEStep(
                    urdf_path=config.ee_urdf_path,
                    motor_names=motor_names,
                )
            )
        elif config.ee_action_space == "ee_pose_delta":
            input_steps.append(
                JointActionToDeltaEEStep(
                    urdf_path=config.ee_urdf_path,
                    motor_names=motor_names,
                )
            )
        else:
            raise ValueError(
                f"Unknown ee_action_space '{config.ee_action_space}'. "
                "Choose from 'joint_pos', 'ee_pose_abs', 'ee_pose_delta'."
            )

    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=effective_stats,
        )
    )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=effective_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
