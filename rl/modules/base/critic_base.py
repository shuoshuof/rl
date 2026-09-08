from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl.networks import EmpiricalNormalization, MLP


class CriticBase(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        cfg: dict,
    ) -> None:
        super().__init__()

        self.obs_groups = cfg["observation_groups"]
        self.obs_group_names = [obs_group["group_name"] for obs_group in self.obs_groups]
        self._resolve_obs_groups(obs)
        self._build_network(cfg["network"])
        self._build_normalizer(obs)

        print(f"Critic: {self}")

    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        # pars obs groups to get input dim
        # Default behavior: 1D observations only
        num_critic_obs = 0
        for group_name in self.obs_group_names:
            assert len(obs[group_name].shape) == 2, "The CriticBase module only supports 1D observations."
            num_critic_obs += obs[group_name].shape[-1]

        self.num_critic_obs = num_critic_obs

    def _build_network(self, network_cfg) -> None:
        mlp_cfg = network_cfg["mlp"]
        self.mlp = MLP(
            input_dim=self.num_critic_obs,
            output_dim=1,
            **mlp_cfg,
        )

    def _build_normalizer(self, obs) -> None:
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
        obs_dict = {
            f"{group_name}_obs": self.obs_normalizers[group_name](obs[group_name])
            for group_name in self.obs_group_names
        }
        return obs_dict

    def forward(self, **kwargs) -> torch.Tensor:
        obs = torch.cat([kwargs[f"{group_name}_obs"] for group_name in self.obs_group_names], dim=-1)
        return self.mlp(obs)
