"""
Base class for RL/IL algorithms used by ft_learner.train().
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader


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
