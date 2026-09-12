from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tensordict import TensorDict

from srl.networks import EmpiricalNormalization


class ActorBase(nn.Module, ABC):
    """Observation handling and construction hooks for an actor.

    Forward returns action means, or means and noise outputs with shape
    [batch, 2, num_actions] when state-dependent standard deviation is enabled.
    """

    def __init__(
        self,
        obs: TensorDict,
        cfg: dict,
        num_actions: int,
        init_noise_std: float = 1.0,
        state_dependent_std: bool = False,
        noise_std_type: str = "scalar",
    ) -> None:
        super().__init__()

        self.output_dim = [2, num_actions] if state_dependent_std else num_actions

        self.obs_groups = cfg["observation_groups"]
        self.obs_group_names = [obs_group["group_name"] for obs_group in self.obs_groups]
        self._build_normalizer(obs)
        self._resolve_obs_groups(obs)
        self._build_network(cfg["network"])
        if state_dependent_std:
            self._init_noise_params(num_actions, init_noise_std, noise_std_type)

        print(f"Actor: {self}")

    @abstractmethod
    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        """Resolve the observation shapes required by the concrete actor."""
        raise NotImplementedError

    @abstractmethod
    def _build_network(self, network_cfg: dict) -> None:
        """Create the actor network from its architecture-specific configuration."""
        raise NotImplementedError

    def _init_noise_params(self, num_actions: int, init_noise_std: float, noise_std_type: str) -> None:
        """Override to initialize the actor's state-dependent noise outputs."""
        raise NotImplementedError("Actors using state_dependent_std must implement _init_noise_params().")

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