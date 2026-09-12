from __future__ import annotations

import torch
from tensordict import TensorDict

from rl.networks import MLP
from rl.registry import CRITICS

from .critic import CriticBase


@CRITICS.register()
class MLPCritic(CriticBase):
    """Critic that concatenates vector observation groups and applies an MLP."""

    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        num_critic_obs = 0
        for group_name in self.obs_group_names:
            assert len(obs[group_name].shape) == 2, "MLPCritic only supports 1D observations."
            num_critic_obs += obs[group_name].shape[-1]
        self.num_critic_obs = num_critic_obs

    def _build_network(self, network_cfg: dict) -> None:
        self.mlp = MLP(
            input_dim=self.num_critic_obs,
            output_dim=1,
            **network_cfg["mlp"],
        )

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        obs = torch.cat([kwargs[f"{group_name}_obs"] for group_name in self.obs_group_names], dim=-1)
        return self.mlp(obs)