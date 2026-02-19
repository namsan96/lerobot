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

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True

Policy options can be overridden from the CLI; they are sent to the server and applied when loading the policy:
    --policy.num_inference_steps=10 --policy.device=cuda
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    fake_robot,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from lerobot.configs import parser as config_parser

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
            rename_map=getattr(config, "rename_map", {}),
            commit_steps=getattr(config, "commit_steps", None),
            policy_cli_overrides=getattr(config, "policy_cli_overrides", None) or [],
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.action_chunk_lock = threading.Lock()
        self.new_action_chunk_ready = threading.Event()
        self.new_action_chunk = None
        self.action_chunk = None

        # Serialize robot access: get_observation() and send_action() use the same port (e.g. Dynamixel).
        self.robot_lock = threading.Lock()

        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # Observation sender: control loop signals with Event; thread reads self.task and does get_observation + send.
        self.observation_requested = threading.Event()
        self.task: str = ""  # set at start of control_loop; observation_sender uses it when signaled

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        self.horizon = self.config.actions_per_chunk
        self.commit_steps = self.config.commit_steps

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()
        self.observation_requested.set()  # wake observation_sender so it can exit

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def observation_sender(self) -> None:
        """Thread: on signal, runs get_observation() and send_observation() so the control loop never blocks on I/O."""
        while self.running:
            self.observation_requested.wait()
            self.observation_requested.clear()
            task = self.task
            if not task:
                continue
            try:
                with self.robot_lock:
                    raw_observation: RawObservation = self.robot.get_observation()
                raw_observation["task"] = task
                observation = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_observation,
                    timestep=0,
                )
                self.send_observation(observation)
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error in observation sender: {e}")

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    raise '??'
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                with self.action_chunk_lock:
                    self.new_action_chunk = timed_actions
                    self.new_action_chunk_ready.set()

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    raise NotImplementedError("Not implemented")
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                if verbose:
                    raise NotImplementedError("Not implemented")
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, timed_action, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""
        with self.robot_lock:
            _performed_action = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )
        if verbose:
            raise NotImplementedError("Not implemented")
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def control_loop_observation(self, verbose: bool = False) -> None:
        """Signal the observation_sender thread to capture and send one observation. Control loop does not block."""
        self.observation_requested.set()

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.task = task
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        chunk_idx = 0

        while self.running:
            control_loop_start = time.perf_counter()

            # Before the first action : wait
            # [0 or latency, S) : do nothing and just execute action
            # S : send observation
            # (S, S+latency) : wait for action chunk and execute action in the previous chunk
            # S+latency : change chunk and set the index to latency
            # if waiting action chunk and action chunk is arrived:
            # update action chunk, set index to the steps elapsed after sending observation
            with self.action_chunk_lock:
                action_chunk = self.action_chunk
            if action_chunk is None:
                self.logger.info("Action chunk ran out, waiting for new action chunk")
                self.control_loop_observation(verbose)
                self.new_action_chunk_ready.wait()
                with self.action_chunk_lock:    
                    action_chunk = self.new_action_chunk
                    self.action_chunk = action_chunk
                    self.new_action_chunk = None
                    self.new_action_chunk_ready.clear()
                chunk_idx = 0

            if chunk_idx == self.commit_steps:
                self.control_loop_observation(verbose)
            elif self.commit_steps < chunk_idx < self.horizon:
                if self.new_action_chunk_ready.is_set():
                    with self.action_chunk_lock:
                        action_chunk = self.action_chunk = self.new_action_chunk
                        self.new_action_chunk = None
                        self.new_action_chunk_ready.clear()
                    delay = chunk_idx - self.commit_steps # even if the action is ready within 1 step (S+1), the first action is skipped
                    chunk_idx = delay
                    self.logger.info(f"Action chunk is ready within delay {delay}steps, executing action")

            # Control robot
            self.control_loop_action(action_chunk[chunk_idx], verbose)

            chunk_idx += 1

            if chunk_idx == self.horizon:  # chunk ran out
                with self.action_chunk_lock:
                    self.action_chunk = None
                chunk_idx = 0

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    # Collect --policy.xxx CLI overrides so they are sent to the server and applied when loading the policy.
    cfg.policy_cli_overrides = config_parser.get_cli_overrides("policy") or []

    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver and observation sender threads...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        observation_sender_thread = threading.Thread(target=client.observation_sender, daemon=True)

        action_receiver_thread.start()
        observation_sender_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            observation_sender_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
