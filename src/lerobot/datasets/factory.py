#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

class _ConcatHfProxy:
    """Minimal proxy that concatenates hf_dataset column access across multiple datasets.

    A column is exposed only if ALL underlying datasets contain it; otherwise the column
    is absent (callers that check `column_names` will use their fallback path).
    """

    def __init__(self, hf_datasets):
        self._datasets = hf_datasets
        common = set(hf_datasets[0].column_names)
        for ds in hf_datasets[1:]:
            common &= set(ds.column_names)
        self.column_names = list(common)

    def __getitem__(self, key):
        return np.concatenate([np.array(ds[key]) for ds in self._datasets])


class _ConcatMeta:
    """Minimal meta object that aggregates episode indices and exposes primary-dataset stats."""

    def __init__(self, datasets):
        self.stats = datasets[0].meta.stats
        self.camera_keys = datasets[0].meta.camera_keys

        offset = 0
        ep_frames = []
        for ds in datasets:
            ep_df = ds.meta.episodes.copy()
            ep_df["dataset_from_index"] = ep_df["dataset_from_index"] + offset
            ep_df["dataset_to_index"] = ep_df["dataset_to_index"] + offset
            ep_frames.append(ep_df)
            offset += ds.num_frames
        self.episodes = pd.concat(ep_frames, ignore_index=True)


class ConcatLeRobotDataset(torch.utils.data.ConcatDataset):
    """Concatenation of multiple LeRobotDataset instances with a unified interface.

    Inherits __len__ and __getitem__ routing from torch.utils.data.ConcatDataset.
    Adds the attributes needed by EpisodeAwareSampler, MetricsTracker, and cfg_utils.
    Stats and camera_keys are taken from the first (primary) dataset.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.num_frames = len(self)
        self.num_episodes = sum(ds.num_episodes for ds in datasets)
        self.meta = _ConcatMeta(datasets)
        self.hf_dataset = _ConcatHfProxy([ds.hf_dataset for ds in datasets])


IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    # Auxiliary keys: policy declares extra data it needs loaded at specific timesteps.
    # Format: {output_batch_key: (source_dataset_feature, delta_indices)}
    aux = getattr(cfg, "auxiliary_delta_indices", None)
    if aux:
        for out_key, (src_key, indices) in aux.items():
            if src_key in ds_meta.features:
                delta_timestamps[out_key] = [i / ds_meta.fps for i in indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def resolve_feature_aliases(cfg: PreTrainedConfig) -> dict[str, str]:
    """Build a feature_aliases dict from cfg.auxiliary_delta_indices.

    Returns a mapping {output_batch_key: source_dataset_feature} for use with LeRobotDataset.
    """
    aux = getattr(cfg, "auxiliary_delta_indices", None)
    if not aux:
        return {}
    return {out_key: src_key for out_key, (src_key, _) in aux.items()}


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        feature_aliases = resolve_feature_aliases(cfg.policy)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                feature_aliases=feature_aliases or None,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                cache_videos=cfg.dataset.cache_videos,
                cache_video_resize=cfg.dataset.cache_video_resize,
                drop_cameras=cfg.dataset.drop_cameras or None,
                image_predecode=cfg.dataset.image_predecode,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    extra_repo_ids = getattr(cfg.dataset, "extra_repo_ids", None)
    if extra_repo_ids:
        extra_datasets = []
        for extra_repo_id in extra_repo_ids:
            extra_meta = LeRobotDatasetMetadata(
                extra_repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
            )
            extra_delta_timestamps = resolve_delta_timestamps(cfg.policy, extra_meta)
            extra_ds = LeRobotDataset(
                extra_repo_id,
                root=cfg.dataset.root,
                delta_timestamps=extra_delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                cache_videos=cfg.dataset.cache_videos,
                cache_video_resize=cfg.dataset.cache_video_resize,
            )
            extra_datasets.append(extra_ds)
            logging.info(f"Concatenating extra dataset: {extra_repo_id} ({extra_ds.num_frames} frames)")
        dataset = ConcatLeRobotDataset([dataset] + extra_datasets)
        logging.info(
            f"Total concatenated dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes"
        )

    return dataset
