from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl.networks import EmpiricalNormalization, MLP


class ActorBase(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        cfg: dict,
        num_actions: int,
        init_noise_std: float=1.0,
        state_dependent_std: bool=False,
        noise_std_type: str="scalar",
    ) -> None:
        super().__init__()

        self.output_dim = [2, num_actions] if state_dependent_std else num_actions

        self.obs_groups = cfg["observation_groups"]
        self.obs_group_names = [obs_group["group_name"] for obs_group in self.obs_groups]
        self._resolve_obs_groups(obs)
        self._build_network(cfg["network"])
        self._build_normalizer(obs)
        if state_dependent_std:
            self._init_noise_params(num_actions, init_noise_std, noise_std_type)

        print(f"Actor: {self}")

    def _resolve_obs_groups(self, obs: TensorDict) -> None:
        # pars obs groups to get input dim
        # Default behavior: 1D observations only
        num_actor_obs = 0
        for group_name in self.obs_group_names:
            assert len(obs[group_name].shape) == 2, "The ActorBase module only supports 1D observations."
            num_actor_obs += obs[group_name].shape[-1]

        self.num_actor_obs = num_actor_obs

    def _init_noise_params(self, num_actions: int, init_noise_std: float, noise_std_type: str) -> None:
        torch.nn.init.zeros_(self.mlp[-2].weight[num_actions:])
        if noise_std_type == "scalar":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], init_noise_std)
        elif noise_std_type == "log":
            torch.nn.init.constant_(self.mlp[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

    def _build_network(self, network_cfg) -> None:
        mlp_cfg = network_cfg["mlp"]
        self.mlp = MLP(
            input_dim=self.num_actor_obs,
            output_dim=self.output_dim,
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
