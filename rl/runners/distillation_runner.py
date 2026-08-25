# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from tensordict import TensorDict

import rl.algorithms as rl_algorithms
import rl.modules as rl_modules
from rl.algorithms import Distillation
from rl.modules import StudentTeacher, StudentTeacherRecurrent
from rl.storage import RolloutStorage
from .on_policy_runner import OnPolicyRunner


class DistillationRunner(OnPolicyRunner):
    """Distillation runner for training and evaluation of teacher-student methods."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        return ["teacher"]

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = getattr(rl_modules, self.policy_cfg.pop("class_name"))
        student_teacher: StudentTeacher | StudentTeacherRecurrent = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "distillation", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        alg_class = getattr(rl_algorithms, self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Set RND configuration to None as it dshoes not apply to distillation
        self.cfg["algorithm"]["rnd_cfg"] = None

        return alg
