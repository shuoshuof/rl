from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl.networks import EmpiricalNormalization, MLP


class CriticBase(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        critic_obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.critic_obs_normalization = critic_obs_normalization
        self.obs_groups = obs_groups["critic"]
        self._resolve_obs_groups(obs, obs_groups)
        self._build_network(**kwargs)
        self._build_normalizer()

        print(f"Critic: {self}")

    def _resolve_obs_groups(self, obs: TensorDict, obs_groups: dict[str, list[str]]) -> None:
        # Default behavior: 1D observations only
        num_critic_obs = 0
        for obs_group in self.obs_groups:
            assert len(obs[obs_group].shape) == 2, "The CriticBase module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        self.num_critic_obs = num_critic_obs

    def _build_network(self, critic_network, **kwargs) -> None:
        mlp_cfg = critic_network.get("mlp", {})
        self.mlp = MLP(
            input_dim=self.num_critic_obs,
            output_dim=1,
            **mlp_cfg,
        )

    def _build_normalizer(self) -> None:
        if self.critic_obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.obs_normalizer = nn.Identity()

    def update_normalization(self, obs) -> None:
        if self.critic_obs_normalization:
            raise NotImplementedError("Critic observation normalization is not implemented")
            self.obs_normalizer.update(obs)

    def resolve_obs(self, obs: TensorDict) -> dict:
        obs_dict = {f"{obs_group}_obs": obs[obs_group] for obs_group in self.obs_groups}
        return obs_dict

    def forward(self, **kwargs) -> torch.Tensor:
        # TODO: add normalization
        obs = torch.cat([kwargs[f"{obs_group}_obs"] for obs_group in self.obs_groups], dim=-1)
        return self.mlp(obs)
