# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin wrapper around the Isaac-GR00T Gr00tN1d6 model for LeRobot integration.

Mirrors the structure of groot_n1.py (GR00TN15) but delegates to the gr00t package
rather than a self-contained port, since N1.6 ships with `gr00t` as an installable
package and its architecture (AlternateVLDiT, updated EagleBackbone) is substantially
different from N1.5.
"""

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

# Compatibility shims for transformers >=4.52 / >=4.53 API removals.
# Eagle-Block2A-2B-v2 processor files reference symbols that were removed or renamed.
import transformers.image_utils as _transformers_image_utils
import transformers.image_processing_utils_fast as _transformers_fast

# VideoInput: removed from image_utils in >=4.52; used as a type annotation only.
if not hasattr(_transformers_image_utils, "VideoInput"):
    from typing import Any, List

    _transformers_image_utils.VideoInput = List[Any]

# make_batched_videos: removed in >=4.53; imported by Eagle fast processor but never called.
if not hasattr(_transformers_image_utils, "make_batched_videos"):
    _transformers_image_utils.make_batched_videos = lambda x: x

# BASE_IMAGE_PROCESSOR_FAST_DOCSTRING*: removed in >=4.53; used only as decorator strings.
if not hasattr(_transformers_fast, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
    _transformers_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
if not hasattr(_transformers_fast, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
    _transformers_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""

# validate_init_kwargs: in >=4.53 returns (unused_kwargs, valid_kwargs) tuple instead of just
# unused_kwargs dict.
#
# Eagle3_VLProcessor has a CUSTOM from_args_and_dict that calls validate_init_kwargs and expects
# a plain dict (old API): `unused_kwargs = cls.validate_init_kwargs(...)` then `kwargs.update(unused_kwargs)`.
#
# Standard transformers 4.53 from_args_and_dict expects the NEW tuple API:
# `unused_kwargs, valid_kwargs = cls.validate_init_kwargs(...)`.
#
# We make the patch caller-aware: return dict only when called from Eagle3's custom from_args_and_dict,
# return full tuple for all other callers (standard transformers processors).
import inspect as _inspect

from transformers.processing_utils import ProcessorMixin as _ProcessorMixin

_orig_validate_init_kwargs = _ProcessorMixin.validate_init_kwargs


@classmethod  # type: ignore[misc]
def _patched_validate_init_kwargs(cls, processor_config, valid_kwargs):
    result = _orig_validate_init_kwargs(processor_config=processor_config, valid_kwargs=valid_kwargs)
    if not isinstance(result, tuple):
        return result
    # Detect if called from Eagle3_VLProcessor's custom from_args_and_dict (uses old dict-only API).
    caller_filename = _inspect.stack()[1].filename
    if "Eagle-Block2A-2B-v2" in caller_filename or "processing_eagle3_vl" in caller_filename:
        return result[0]  # old API: return just unused_kwargs dict
    return result  # new API: return (unused_kwargs, valid_kwargs) tuple


_ProcessorMixin.validate_init_kwargs = _patched_validate_init_kwargs

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6


class GR00TN1d6:
    """Facade that loads a Gr00tN1d6 pretrained model and applies tune flags.

    Usage mirrors GR00TN15.from_pretrained:

        model = GR00TN1d6.from_pretrained(
            pretrained_model_name_or_path="nvidia/GR00T-N1.6-3B",
            tune_llm=False,
            tune_visual=False,
            tune_top_llm_layers=4,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
        )
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_top_llm_layers: int = 4,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vlln: bool = True,
        **kwargs,
    ) -> Gr00tN1d6:
        """Load a pretrained Gr00tN1d6 and apply tune flags.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            tune_llm: Fine-tune the full LLM backbone.
            tune_visual: Fine-tune the visual encoder.
            tune_top_llm_layers: Number of top LLM layers to unfreeze even when
                tune_llm=False.  Matches the N1.6 default of 4.
            tune_projector: Fine-tune the action-head projector (state/action encoders).
            tune_diffusion_model: Fine-tune the DiT diffusion model.
            tune_vlln: Fine-tune the VL LayerNorm in the action head.
            **kwargs: Forwarded to Gr00tN1d6.from_pretrained.

        Returns:
            Loaded Gr00tN1d6 model with tune flags applied.
        """
        print(f"Loading pretrained GR00T-N1.6 from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}  (top layers: {tune_top_llm_layers})")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Tune action head VL LayerNorm: {tune_vlln}")

        # Resolve to local path (HF cache) so Gr00tN1d6.from_pretrained gets a local dir.
        try:
            local_model_path = snapshot_download(
                pretrained_model_name_or_path, repo_type="model"
            )
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found in HuggingFace Hub. "
                f"Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        model: Gr00tN1d6 = Gr00tN1d6.from_pretrained(local_model_path, **kwargs)

        # Apply tune flags — N1.6 backbone takes tune_top_llm_layers as a third argument.
        model.backbone.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        model.action_head.set_trainable_parameters(tune_projector, tune_diffusion_model, tune_vlln)

        return model
