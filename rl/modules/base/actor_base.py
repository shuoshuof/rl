from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl.networks import EmpiricalNormalization, MLP


class ActorBase(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool,
        init_noise_std: float=1.0,
        state_dependent_std: bool=False,
        noise_std_type: str="scalar",
        **kwargs,
    ) -> None:
        super().__init__()

        self.output_dim = [2, num_actions] if state_dependent_std else num_actions
        self.actor_obs_normalization = actor_obs_normalization

        self.obs_groups = obs_groups["actor"]
        self._resolve_obs_groups(obs)
        self._build_network(**kwargs)
        self._build_normalizer()
        if state_dependent_std:
            self._init_noise_params(num_actions, init_noise_std, noise_std_type)

        print(f"Actor: {self}")

    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        # pars obs groups to get input dim
        # Default behavior: 1D observations only
        num_actor_obs = 0
        for obs_group in self.obs_groups:
            assert len(obs[obs_group].shape) == 2, "The ActorBase module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        self.num_actor_obs = num_actor_obs

    def _init_noise_params(self, num_actions: int, init_noise_std: float, noise_std_type: str) -> None:
        torch.nn.init.zeros_(self.mlp[-2].weight[num_actions:])
        if noise_std_type == "scalar":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], init_noise_std)
        elif noise_std_type == "log":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

    def _build_network(self, actor_network, **kwargs) -> None:
        mlp_cfg = actor_network.get("mlp", {})
        self.mlp = MLP(
            input_dim=self.num_actor_obs,
            **mlp_cfg,
            output_dim=self.output_dim,
        )

    def _build_normalizer(self) -> None:
        if self.actor_obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.obs_normalizer = nn.Identity()

    def update_normalization(self, obs: torch.Tensor) -> None:
        if self.actor_obs_normalization:
            raise NotImplementedError("Actor observation normalization is not implemented")
            self.obs_normalizer.update(obs)

    def resolve_obs(self, obs: TensorDict) -> dict:
        obs_dict = {f"{obs_group}_obs": obs[obs_group] for obs_group in self.obs_groups}
        return obs_dict

    def forward(self, **kwargs) -> torch.Tensor:
        # TODO: add normalization
        obs = torch.cat([kwargs[f"{obs_group}_obs"] for obs_group in self.obs_groups], dim=-1)
        return self.mlp(obs)
