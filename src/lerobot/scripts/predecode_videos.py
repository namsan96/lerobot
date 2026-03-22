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
"""Pre-decode all video frames to PNG files for faster loading during training.

Usage:
    python -m lerobot.scripts.predecode_videos \
        --dataset_root /path/to/dataset \
        --repo_id lerobot/my_dataset \
        --num_workers 4
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import PIL.Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from lerobot.datasets.video_utils import decode_video_frames

logger = logging.getLogger(__name__)


def _decode_episode(
    ep_idx: int,
    total_episodes: int,
    ep: dict,
    vid_key: str,
    dataset_root: str,
    fps: int,
    video_path: str,
    size: tuple[int, int] | None = None,
) -> str:
    """Top-level worker function for ProcessPoolExecutor (must be picklable).

    Returns a status string for logging.
    """
    dataset_root = Path(dataset_root)
    video_path = Path(video_path)

    n_frames = ep["dataset_to_index"] - ep["dataset_from_index"]

    # Compute episode image dir from the first frame path
    first_frame_path = dataset_root / DEFAULT_IMAGE_PATH.format(
        image_key=vid_key,
        episode_index=ep_idx,
        frame_index=0,
    )
    episode_dir = first_frame_path.parent

    # Skip if all frames already exist
    if episode_dir.exists():
        existing = list(episode_dir.glob("frame-*.png"))
        if len(existing) == n_frames:
            return f"episode {ep_idx}/{total_episodes - 1}: skipped (already decoded)"

    from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
    abs_timestamps = [from_timestamp + i / fps for i in range(n_frames)]

    frames = decode_video_frames(video_path, abs_timestamps, tolerance_s=0.04)
    # frames: (n_frames, C, H, W) uint8

    episode_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        frame_path = episode_dir / f"frame-{i:06d}.png"
        img = PIL.Image.fromarray(frames[i].permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy())
        if size is not None:
            img = img.resize((size[1], size[0]), PIL.Image.LANCZOS)  # PIL uses (W, H)
        img.save(frame_path)

    return f"episode {ep_idx}/{total_episodes - 1}: decoded {n_frames} frames"


def predecode_videos(dataset_root: str | Path, repo_id: str, num_workers: int = 4, size: int | tuple[int, int] | None = None) -> None:
    if isinstance(size, int):
        size = (size, size)
    dataset_root = Path(dataset_root)

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)

    total_episodes = meta.total_episodes
    fps = meta.fps
    video_keys = meta.video_keys

    if not video_keys:
        logger.info("No video keys found in dataset metadata. Nothing to decode.")
        return

    # Build list of (ep_idx, vid_key) tasks
    tasks = []
    for ep_idx in range(total_episodes):
        ep = meta.episodes[ep_idx]
        for vid_key in video_keys:
            video_path = dataset_root / meta.get_video_file_path(ep_idx, vid_key)
            tasks.append((ep_idx, ep, vid_key, video_path))

    num_workers = min(num_workers, len(tasks))
    logger.info(
        "Pre-decoding %d episode×video-key combinations with %d workers.",
        len(tasks),
        num_workers,
    )

    if num_workers <= 1:
        for ep_idx, ep, vid_key, video_path in tasks:
            result = _decode_episode(
                ep_idx=ep_idx,
                total_episodes=total_episodes,
                ep=ep,
                vid_key=vid_key,
                dataset_root=str(dataset_root),
                fps=fps,
                video_path=str(video_path),
                size=size,
            )
            logger.info("%s [vid_key=%s]", result, vid_key)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for ep_idx, ep, vid_key, video_path in tasks:
                future = executor.submit(
                    _decode_episode,
                    ep_idx=ep_idx,
                    total_episodes=total_episodes,
                    ep=ep,
                    vid_key=vid_key,
                    dataset_root=str(dataset_root),
                    fps=fps,
                    video_path=str(video_path),
                    size=size,
                )
                futures[future] = (ep_idx, vid_key)

            for future in as_completed(futures):
                ep_idx, vid_key = futures[future]
                try:
                    result = future.result()
                    logger.info("%s [vid_key=%s]", result, vid_key)
                except Exception as exc:
                    logger.error(
                        "episode %d vid_key=%s raised an exception: %s",
                        ep_idx,
                        vid_key,
                        exc,
                    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-decode all video frames to PNG files for faster dataset loading."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=str,
        help="Local root directory of the dataset.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        type=str,
        help="HuggingFace repo ID of the dataset (e.g. lerobot/my_dataset).",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of parallel worker processes (default: 4).",
    )
    args = parser.parse_args()

    predecode_videos(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
