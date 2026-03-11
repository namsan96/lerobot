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
Example (diffusion policy with DINOv3, online RL weight hot-swap from ft_learner):

    python -m lerobot.async_inference.rtc_server \\
        --pretrained_name_or_path=tw_outputs/diffusion/pretrained_model \\
        --policy_type=diffusion \\
        --host=0.0.0.0 \\
        --port=8080 \\
        --device=cuda \\
        --fps=30 \\
        --weights_watch_dir=outputs/rl/pick_and_place \\
        --weights_check_interval=5.0 \\
        --policy.dinov3_hub_repo=facebookresearch/dinov2 \\
        --policy.dinov3_hub_weights=dinov2_vits14
"""

import logging
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Lock protecting policy weights during hot-swap from ft_learner
        self._policy_lock = threading.Lock()
        self._weights_watcher_thread: threading.Thread | None = None

        # Session-specific config (set by SendPolicyInstructions)
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.commit_steps = None
        self.task: str = ""
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        # Last action chunk for action-conditioned inference (set when commit_steps is configured)
        self.last_action_chunk: torch.Tensor | None = None

        # Load policy at server startup
        self.device = config.device
        self.policy_type = config.policy_type
        cli_overrides = getattr(config, "policy_cli_overrides", None) or []
        policy_class = get_policy_class(config.policy_type)
        self.logger.info(
            f"Loading policy '{config.policy_type}' from '{config.pretrained_name_or_path}' on {config.device}"
            + (f" with overrides {cli_overrides}" if cli_overrides else "")
        )
        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(
            config.pretrained_name_or_path,
            cli_overrides=cli_overrides,
        )
        self.policy.to(config.device)
        self.policy.eval()
        self.logger.info(f"Policy loaded in {time.perf_counter() - start:.2f}s")

        # Build preprocessor/postprocessor with empty rename_map; updated per-session in SendPolicyInstructions
        device_override = {"device": config.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=config.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        # Mixed-precision inference setup (no GradScaler needed — inference only)
        mp = config.mixed_precision
        _dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float16)
        self._autocast_kwargs = dict(device_type=config.device.split(":")[0], dtype=_dtype, enabled=mp is not None)

        if config.weights_watch_dir is not None:
            self._start_weights_watcher()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        self.last_action_chunk = None

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive session-specific config from the robot client.

        Policy is already loaded at server startup; this only updates per-session
        parameters: lerobot_features, actions_per_chunk, commit_steps, rename_map.
        """
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self.logger.info(
            f"Receiving session config from {client_id} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Commit steps: {policy_specs.commit_steps} | "
            f"Task: '{policy_specs.task}'"
        )

        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.commit_steps = policy_specs.commit_steps
        self.task = policy_specs.task or ""

        # Rebuild preprocessor with client-provided rename_map (cheap — no policy reload)
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.config.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        self.observation_queue.put(timed_observation)
        # if not self._enqueue_observation(
        #     timed_observation  # wrapping a RawObservation
        # ):
        #     self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get()  # block until observation is enqueued (no timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )
            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        # Policies that use observation queues (e.g. Diffusion, VQ-BeT, TD-MPC) expect
        # _queues to be filled before predict_action_chunk (which stacks from _queues).
        batch = dict(observation)
        # Inference uses observation only; drop action if preprocessor left it in (e.g. pipeline shape).
        if ACTION in batch:
            batch.pop(ACTION)
        if getattr(self.policy, "_queues", None) is not None:
            if getattr(self.policy.config, "image_features", None):
                batch[OBS_IMAGES] = torch.stack(
                    [batch[k] for k in self.policy.config.image_features], dim=-4
                )
            self.policy._queues = populate_queues(self.policy._queues, batch)

        kwargs = {}
        if self.commit_steps is not None and self.last_action_chunk is not None:
            kwargs["action_cond"] = self.last_action_chunk[:, self.commit_steps:, :]

        with self._policy_lock, torch.amp.autocast(**self._autocast_kwargs):
            if self.config.use_pt_act_steps:
                chunk = self.policy.predict_action_chunk(batch, **kwargs)
            else:
                chunk = self.policy.predict_action_chunk(batch, full_length=True, **kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        chunk = chunk[:, : self.actions_per_chunk, :]

        # Save for action-conditioned inference on next call
        if self.commit_steps is not None:
            self.last_action_chunk = chunk

        return chunk

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        raw_obs = observation_t.get_observation()
        # Fall back to the session-level task if the observation doesn't carry one
        if "task" not in raw_obs and self.task:
            raw_obs = dict(raw_obs)
            raw_obs["task"] = self.task
        observation: Observation = raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    # ------------------------------------------------------------------
    # Weight hot-swap (ft_learner integration)
    # ------------------------------------------------------------------

    def _start_weights_watcher(self) -> None:
        """Start a background thread that polls latest_weights.pt and hot-swaps policy weights."""
        if self._weights_watcher_thread is not None and self._weights_watcher_thread.is_alive():
            return  # already running
        self._weights_watcher_thread = threading.Thread(
            target=self._weights_watcher_loop, daemon=True, name="weights-watcher"
        )
        self._weights_watcher_thread.start()
        self.logger.info(
            f"[WeightsWatcher] Started — watching {self.config.weights_watch_dir}/latest_weights.pt "
            f"every {self.config.weights_check_interval}s"
        )

    def _weights_watcher_loop(self) -> None:
        """Poll latest_weights.pt; load and hot-swap when mtime changes."""
        weights_path = Path(self.config.weights_watch_dir) / "latest_weights.pt"
        last_mtime: float | None = None

        while not self.shutdown_event.is_set():
            try:
                if weights_path.exists():
                    mtime = weights_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                        with self._policy_lock:
                            self.policy.load_state_dict(state_dict, strict=False)
                            if hasattr(self.policy, "_q_initialized"):
                                self.policy._q_initialized = True
                        self.logger.info(f"[WeightsWatcher] Weights hot-swapped from {weights_path}")
            except Exception as e:
                self.logger.warning(f"[WeightsWatcher] Failed to load weights: {e}")

            self.shutdown_event.wait(timeout=self.config.weights_check_interval)

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    # Apply policy overrides collected by __main__ (--policy.xxx stripped from argv before parse)
    if getattr(serve, "_policy_cli_overrides", None) is not None:
        cfg.policy_cli_overrides = getattr(serve, "_policy_cli_overrides")

    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    import sys

    from lerobot.configs import parser as config_parser

    # Collect --policy.xxx and remove from argv so draccus does not reject them (PolicyServerConfig has no policy field).
    serve._policy_cli_overrides = config_parser.get_cli_overrides("policy") or []
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--policy.")]
    serve()
