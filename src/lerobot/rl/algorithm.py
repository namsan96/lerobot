"""
Base class for RL/IL algorithms used by ft_learner.train().
"""

import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass

import draccus
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class AlgorithmConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore[misc]
    """
    Base config for all fine-tuning algorithms.

    Subclasses register themselves with::

        @AlgorithmConfig.register_subclass("my_alg")
        @dataclass
        class MyAlgConfig(AlgorithmConfig):
            ...

    This enables the CLI pattern ``--alg.type=my_alg --alg.lr=1e-4``.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)  # type: ignore[return-value]

    @abc.abstractmethod
    def make_algorithm(self, policy: nn.Module) -> "Algorithm":
        """Instantiate the algorithm from this config and the given policy."""
        ...


class Algorithm(ABC):
    """
    Owns the optimizer(s) for a given policy. Responsible for one full outer iteration.

    Subclasses implement different RL / IL algorithms (BC, SAC, TD3, REINFORCE, ...).
    The policy is created externally via make_policy() and passed in, so the same
    policy instance is shared between the Algorithm and the training loop (for weight push).

    The algorithm owns its own data iterator and pulls as many batches as it needs per call.
    """

    def __init__(self, policy: nn.Module) -> None:
        self._policy = policy
        self.preprocessor = None  # set externally after construction
        self.postprocessor = None  # set externally after construction

    @property
    def policy(self) -> nn.Module:
        return self._policy

    @abstractmethod
    def update(self, loader: DataLoader, itr: int) -> dict:
        """
        Run one full outer iteration (however many gradient steps the algorithm needs).

        Args:
            loader: DataLoader for the current dataset (may change across calls on reload).
            itr: Outer iteration index.

        Returns:
            Info dict to be logged.
        """
        ...
