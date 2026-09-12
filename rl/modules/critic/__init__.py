"""Critic interfaces and registered implementations."""

from .critic import CriticBase
from .mlp_critic import MLPCritic

__all__ = ["CriticBase", "MLPCritic"]