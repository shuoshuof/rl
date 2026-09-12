# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict


class RolloutStorage:
    """Storage for transitions collected by a feedforward on-policy algorithm."""

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    class Batch:
        """A mini-batch of rollout data for feedforward PPO updates."""

        def __init__(
            self,
            observations: TensorDict,
            actions: torch.Tensor,
            values: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_actions_log_prob: torch.Tensor,
            old_mu: torch.Tensor,
            old_sigma: torch.Tensor,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict = observations
            """Batch of observations."""

            self.actions: torch.Tensor = actions
            """Batch of actions."""

            self.values: torch.Tensor = values
            """Batch of value estimates recorded during rollout."""

            self.advantages: torch.Tensor = advantages
            """Batch of advantage estimates."""

            self.returns: torch.Tensor = returns
            """Batch of return targets."""

            self.old_actions_log_prob: torch.Tensor = old_actions_log_prob
            """Batch of action log probabilities under the rollout policy."""

            self.old_mu: torch.Tensor = old_mu
            """Batch of means of the rollout action distribution."""

            self.old_sigma: torch.Tensor = old_sigma
            """Batch of standard deviations of the rollout action distribution."""

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Yield the mini-batch
                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_mu=old_mu[batch_idx],
                    old_sigma=old_sigma[batch_idx],
                )
