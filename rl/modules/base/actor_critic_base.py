from __future__ import annotations
from abc import ABC

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import NoReturn

from .actor_base import ActorBase
from .critic_base import CriticBase


class ActorCriticBase(nn.Module, ABC):
    actor: ActorBase
    critic: CriticBase

    def __init__(
        self,
        num_actions: int,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self._init_noise_params(num_actions, init_noise_std, noise_std_type)

    def _init_noise_params(
        self,
        num_actions: int,
        init_noise_std: float,
        noise_std_type: str,
    ) -> None:
        # Action noise
        self.noise_std_type = noise_std_type
        if not self.state_dependent_std:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, actor_obs: dict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(**actor_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(**actor_obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor.resolve_obs(obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor.resolve_obs(obs)
        if self.state_dependent_std:
            return self.actor(**actor_obs)[..., 0, :]
        else:
            return self.actor(**actor_obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        critic_obs = self.critic.resolve_obs(obs)
        return self.critic(**critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine whether to restore optimizer and iteration state.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    def compile(self, *args, **kwargs) -> None:
        self.actor.compile(*args, **kwargs)
        self.critic.compile(*args, **kwargs)
