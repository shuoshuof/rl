from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tensordict import TensorDict

from srl.networks import EmpiricalNormalization


class CriticBase(nn.Module, ABC):
    """Observation handling and construction hooks for a critic returning [batch, 1] values."""

    def __init__(
        self,
        obs: TensorDict,
        cfg: dict,
    ) -> None:
        super().__init__()

        self.obs_groups = cfg["observation_groups"]
        self.obs_group_names = [obs_group["group_name"] for obs_group in self.obs_groups]
        self._build_normalizer(obs)
        self._resolve_obs_groups(obs)
        self._build_network(cfg["network"])

        print(f"Critic: {self}")

    @abstractmethod
    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        """Resolve the observation shapes required by the concrete critic."""
        raise NotImplementedError

    @abstractmethod
    def _build_network(self, network_cfg: dict) -> None:
        """Create the critic network from its architecture-specific configuration."""
        raise NotImplementedError

    def _build_normalizer(self, obs: TensorDict) -> None:
        self.obs_normalizers = nn.ModuleDict()
        for obs_group in self.obs_groups:
            group_name = obs_group["group_name"]
            if obs_group.get("normalize", False):
                obs_shape = obs[group_name].shape[1:]
                self.obs_normalizers[group_name] = EmpiricalNormalization(obs_shape)
            else:
                self.obs_normalizers[group_name] = nn.Identity()

    def update_normalization(self, obs: TensorDict) -> None:
        for group_name, normalizer in self.obs_normalizers.items():
            if isinstance(normalizer, EmpiricalNormalization):
                normalizer.update(obs[group_name])

    def resolve_obs(self, obs: TensorDict) -> dict:
        return {
            f"{group_name}_obs": self.obs_normalizers[group_name](obs[group_name])
            for group_name in self.obs_group_names
        }

    @abstractmethod
    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError