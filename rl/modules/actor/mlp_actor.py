from __future__ import annotations

import torch
from tensordict import TensorDict

from rl.networks import MLP
from rl.registry import ACTORS

from .actor import ActorBase


@ACTORS.register()
class MLPActor(ActorBase):
    """Actor that concatenates vector observation groups and applies an MLP."""

    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        num_actor_obs = 0
        for group_name in self.obs_group_names:
            assert len(obs[group_name].shape) == 2, "MLPActor only supports 1D observations."
            num_actor_obs += obs[group_name].shape[-1]
        self.num_actor_obs = num_actor_obs

    def _build_network(self, network_cfg: dict) -> None:
        self.mlp = MLP(
            input_dim=self.num_actor_obs,
            output_dim=self.output_dim,
            **network_cfg["mlp"],
        )

    def _init_noise_params(self, num_actions: int, init_noise_std: float, noise_std_type: str) -> None:
        torch.nn.init.zeros_(self.mlp[-2].weight[num_actions:])
        if noise_std_type == "scalar":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], init_noise_std)
        elif noise_std_type == "log":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        obs = torch.cat([kwargs[f"{group_name}_obs"] for group_name in self.obs_group_names], dim=-1)
        return self.mlp(obs)