# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import torch
import warnings
from tensordict import TensorDict
from typing import Any, Callable


def get_param(param: Any, idx: int) -> Any:
    """Get a parameter for the given index.

    Args:
        param: Parameter or list/tuple of parameters.
        idx: Index to get the parameter for.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
        "silu": torch.nn.SiLU(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve the optimizer from the name.

    Args:
        optimizer_name: Name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def string_to_callable(name: str) -> Callable:
    """Resolve the module and function names to return the function.

    Args:
        name: Function name. The format should be 'module:attribute_name'.

    Returns:
        The function loaded from the module.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # Check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as err:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}'."
        )
        raise ValueError(msg) from err


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], required_sets: list[str]
) -> dict[str, list[str]]:
    """Validate observation groups and the sets required by the algorithm.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like:
        {
            "actor": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"]
        }

    This means that the 'actor' observation set will contain the observations "group_1" and "group_2" and the 'critic'
    observation set will contain the observations "group_1" and "group_3". This function will check that all the
    observations in the 'actor' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Dictionary mapping observation sets to lists of observation groups.
        required_sets: Observation set names required by the algorithm and its extensions.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
        ValueError: If a required observation set is missing.
    """
    # Check if obs_groups dictionary is empty
    if len(obs_groups) == 0:
        warnings.warn(
            "The observation configuration dictionary 'obs_groups' is empty and thus likely not configured. Consider"
            " configuring the 'obs_groups' dictionary explicitly"
        )
    else:
        # Check all observation sets for valid observation groups
        for set_name, groups in obs_groups.items():
            # Check if the list is empty
            if len(groups) == 0:
                raise ValueError(f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list.")
            # Check groups exist inside the observations from the environment
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                        f" environment. Available observations from the environment: {list(obs.keys())}"
                    )

    missing_sets = [set_name for set_name in required_sets if set_name not in obs_groups]
    if missing_sets:
        raise ValueError(f"Missing required observation sets: {missing_sets}.")

    # Print the final parsed observation sets
    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups
