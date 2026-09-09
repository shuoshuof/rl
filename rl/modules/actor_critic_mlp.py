from __future__ import annotations

from tensordict import TensorDict

from .base import ActorBase, ActorCriticBase, CriticBase


class ActorCriticMLP(ActorCriticBase):
    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        actor_cfg: dict,
        critic_cfg: dict,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__(
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
        )
        self.actor = ActorBase(
            obs=obs,
            cfg=actor_cfg,
            num_actions=num_actions,
            state_dependent_std=state_dependent_std,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        self.critic = CriticBase(
            obs=obs,
            cfg=critic_cfg
        )
