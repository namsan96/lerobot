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
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
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
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
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
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility


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
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            rename_map=getattr(config, "rename_map", {}),
            commit_steps=config.commit_steps if getattr(config, "use_action_cond", False) else None,
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

        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # Observation sender: control loop captures obs each iteration, stores here, then signals sender.
        # No lock needed: event.set() happens-after the write, event.wait() happens-before the read.
        self.latest_raw_obs: RawObservation | None = None
        self.observation_requested = threading.Event()
        self.task: str = ""  # set at start of control_loop; observation_sender uses it when signaled

        # Dataset recording (optional)
        self.dataset: LeRobotDataset | None = None
        if config.dataset_repo_id is not None:
            num_cameras = len(self.robot.cameras) if hasattr(self.robot, "cameras") else 0
            if config.dataset_resume:
                self.dataset = LeRobotDataset(
                    config.dataset_repo_id,
                    root=config.dataset_root,
                )
                if config.dataset_num_image_writer_processes or config.dataset_num_image_writer_threads_per_camera:
                    self.dataset.start_image_writer(
                        num_processes=config.dataset_num_image_writer_processes,
                        num_threads=config.dataset_num_image_writer_threads_per_camera * num_cameras,
                    )
            else:
                dataset_features = self._make_dataset_features(config.dataset_video)
                self.dataset = LeRobotDataset.create(
                    config.dataset_repo_id,
                    config.fps,
                    root=config.dataset_root,
                    robot_type=self.robot.name,
                    features=dataset_features,
                    use_videos=config.dataset_video,
                    image_writer_processes=config.dataset_num_image_writer_processes,
                    image_writer_threads=config.dataset_num_image_writer_threads_per_camera * num_cameras,
                )

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Keyboard listener for episode control (right=next, left=rerecord, esc=stop)
        self.listener, self.events = init_keyboard_listener()

        self.logger.info("Robot connected and ready")

        # e.g. downsample = 4 => a3, a7, ...
        # chunk 7 => horizon 1 / chunk 8 => horizon 2
        self.horizon = self.config.actions_per_chunk //self.config.downsample
        self.commit_steps = self.config.commit_steps

    def _make_dataset_features(self, use_videos: bool) -> dict:
        obs_features = hw_to_dataset_features(self.robot.observation_features, OBS_STR, use_video=use_videos)
        action_features = hw_to_dataset_features(self.robot.action_features, ACTION, use_video=use_videos)
        reward_feature = {"reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]}}
        terminated_feature = {"terminated": {"dtype": "float32", "shape": (1,), "names": ["terminated"]}}
        debug_features = {
            "debug.elapsed_ms": {"dtype": "float32", "shape": (1,), "names": ["elapsed_ms"]},
            "debug.chunk_idx": {"dtype": "float32", "shape": (1,), "names": ["chunk_idx"]},
        }
        return combine_feature_dicts(obs_features, action_features, reward_feature, terminated_feature, debug_features)

    def _restart_keyboard_listener(self):
        self.listener, self.events = init_keyboard_listener()

    def _wait_for_enter(self, prompt: str = ""):
        import sys, termios
        if self.listener is not None:
            self.listener.stop()
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            input(prompt)
        finally:
            self._restart_keyboard_listener()

    def _prompt_terminated(self) -> float:
        import sys, termios
        if self.listener is not None:
            self.listener.stop()
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            while True:
                val = input("Terminated? (0=False, 1=True): ").strip()
                if val in ("0", "1"):
                    return float(val)
                print("Please enter 0 or 1.")
        finally:
            self._restart_keyboard_listener()


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

            self.logger.info("Sending session config to policy server")

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

        if self.dataset is not None:
            # Safety net: save any frames buffered by an unexpected shutdown (e.g. SIGINT mid-episode)
            if self.dataset.episode_buffer is not None and self.dataset.episode_buffer["size"] > 0:
                self.dataset.save_episode()
                self.logger.info("Saved partial episode on shutdown")
            if self.dataset.image_writer is not None:
                self.dataset.image_writer.stop()

        if self.listener is not None:
            self.listener.stop()
        
        self.robot.go_to_home()

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
        """Thread: on signal from control loop, reads the already-captured observation and sends it.
        The control loop writes latest_raw_obs then sets observation_requested, so no lock is needed."""
        while self.running:
            self.observation_requested.wait()
            self.observation_requested.clear()
            raw_obs = self.latest_raw_obs
            task = self.task
            if raw_obs is None or not task:
                continue
            try:
                obs_with_task = dict(raw_obs)
                obs_with_task["task"] = task
                observation = TimedObservation(
                    timestamp=time.time(),
                    observation=obs_with_task,
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

                # DOWNSAMPLING
                ds = self.config.downsample
                timed_actions = timed_actions[ds-1::ds]

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
        """Execute action on robot. Returns the action dict sent (pre-clip, like lerobot_record)."""
        action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
        self.robot.send_action(action_dict)
        if verbose:
            raise NotImplementedError("Not implemented")

        return action_dict

    def control_loop_observation(self, verbose: bool = False) -> None:
        """Signal the observation_sender thread to capture and send one observation. Control loop does not block."""
        self.observation_requested.set()

    def _reset_chunk_state(self):
        """Reset action chunk state between episodes."""
        with self.action_chunk_lock:
            self.action_chunk = None
            self.new_action_chunk = None
        self.new_action_chunk_ready.clear()

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations.

        Calls robot.get_observation() every iteration (like lerobot_record). The captured
        observation is stored in self.latest_raw_obs before signaling the observation_sender
        thread, so no lock is needed between them.

        Keyboard controls (when dataset recording is active):
          right arrow — finish episode and save, start next
          left arrow  — discard episode buffer and rerecord
          escape      — save current episode and stop
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.task = task
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        chunk_idx = 0

        self.robot.go_to_home()
        self._wait_for_enter("Press Enter to start first episode...")

        # Outer episode loop
        while self.running:
            # --- inner per-step loop (one episode) ---
            while self.running and not self.events["exit_early"]:
                control_loop_start = time.perf_counter()

                # Capture observation every iteration (like lerobot_record)
                raw_obs: RawObservation = self.robot.get_observation()
                # Write before signaling — observation_sender reads after wait(), no lock needed
                self.latest_raw_obs = raw_obs

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
                        delay = chunk_idx - self.commit_steps  # even if the action is ready within 1 step (S+1), the first action is skipped
                        chunk_idx = delay
                        self.logger.info(f"Action chunk is ready within delay {delay}steps, executing action")

                # Control robot; get back action dict for dataset saving
                action_dict = self.control_loop_action(action_chunk[chunk_idx], verbose)
                _performed_action = action_dict

                # Save frame to dataset (if recording); reward set per-step via '1' key
                if self.dataset is not None:
                    obs_frame = build_dataset_frame(self.dataset.features, raw_obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(self.dataset.features, action_dict, prefix=ACTION)
                    elapsed_ms = (time.perf_counter() - control_loop_start) * 1000
                    step_reward = 1.0 if self.events["reward_1"] else 0.0
                    self.events["reward_1"] = False
                    self.dataset.add_frame({
                        **obs_frame,
                        **action_frame,
                        "task": task,
                        "reward": np.array([step_reward], dtype=np.float32),
                        "terminated": np.array([0.0], dtype=np.float32),
                        "debug.elapsed_ms": np.array([elapsed_ms], dtype=np.float32),
                        "debug.chunk_idx": np.array([chunk_idx], dtype=np.float32),
                    })

                chunk_idx += 1

                if chunk_idx == self.horizon:  # chunk ran out
                    with self.action_chunk_lock:
                        self.action_chunk = None
                    chunk_idx = 0

                self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
                # Dynamically adjust sleep time to maintain the desired control frequency
                time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

            # --- episode boundary ---
            if self.events["stop_recording"]:
                self.shutdown_event.set()
            self.robot.go_to_home()

            if self.dataset is not None:
                print(f"Current episode index: {self.dataset.num_episodes}")
                if self.events["rerecord_episode"]:
                    self.logger.info("Left arrow: discarding episode buffer, will rerecord")
                    self.dataset.clear_episode_buffer()
                    self._wait_for_enter("Enter to start next episode")
                else:
                    terminated = self._prompt_terminated()
                    self.dataset.episode_buffer["terminated"][-1] = np.array([terminated], dtype=np.float32)
                    self.dataset.save_episode()
                    # Close data parquet writer so ft_learner (separate process) can read the file.
                    # _writer_closed_for_reading tells _save_episode_data to open a new file next episode.
                    self.dataset._close_writer()
                    self.dataset._writer_closed_for_reading = True
                    self.logger.info(f"Episode {self.dataset.num_episodes} saved with terminated={bool(terminated)}")
                    self._wait_for_enter("Enter to start next episode")
            else:
                self._wait_for_enter("Enter to restart")

            # Reset flags and chunk state for the next episode
            self.events["exit_early"] = False
            self.events["rerecord_episode"] = False
            self._reset_chunk_state()
            chunk_idx = 0

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
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
