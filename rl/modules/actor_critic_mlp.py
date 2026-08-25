from __future__ import annotations

from tensordict import TensorDict

from .base import ActorBase, ActorCriticBase, CriticBase


class ActorCriticMLP(ActorCriticBase):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )
        self.actor = ActorBase(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            state_dependent_std=state_dependent_std,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            actor_obs_normalization=actor_obs_normalization,
            **kwargs,
        )
        self.critic = CriticBase(
            obs=obs,
            obs_groups=obs_groups,
            critic_obs_normalization=critic_obs_normalization,
            **kwargs,
        )
