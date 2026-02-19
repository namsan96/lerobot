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

"""FakeRobot: no-op robot for offline testing. get_observation and send_action always return zeros."""

from functools import cached_property
from typing import Any

import torch

from ..robot import Robot
from .config_fake_robot import FakeRobotConfig


# Same joint keys as SO100Follower so policies/datasets are compatible
FAKE_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class FakeRobot(Robot):
    """
    Fake robot for offline testing. No hardware; connect/disconnect are no-ops.
    get_observation() returns zeros for all state and image keys.
    send_action() returns the same action (or zeros) without sending anywhere.
    """

    config_class = FakeRobotConfig
    name = "fake_robot"

    def __init__(self, config: FakeRobotConfig):
        super().__init__(config)
        self.config = config

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        out = {k: float for k in FAKE_STATE_KEYS}
        for cam_key, cam_config in self.config.cameras.items():
            out[cam_key] = (cam_config.height, cam_config.width, 3)
        return out

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in FAKE_STATE_KEYS}

    @property
    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for k in FAKE_STATE_KEYS:
            obs[k] = 0.0
        for cam_key, cam_config in self.config.cameras.items():
            h, w = cam_config.height, cam_config.width
            obs[cam_key] = torch.zeros((h, w, 3), dtype=torch.uint8)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Return zeros so caller sees "action applied" as zeros (no real effect)
        return {k: 0.0 for k in self.action_features}

    def disconnect(self) -> None:
        self._connected = False
