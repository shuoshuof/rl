# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-layer perceptron.

    Hidden layers each include an activation. An output layer is appended only when ``output_dim`` is specified.
    The optional last activation is appended after all other layers.

    It provides additional conveniences:
    - Without ``input_dim``, the first layer infers its input width using LazyLinear.
    - With an explicit input dimension, hidden dimensions of ``-1`` use that width.
    - Tuple/list output dimensions reshape the last dimension, preserving leading dimensions.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] | list[int],
        input_dim: int | None = None,
        output_dim: int | tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            hidden_dims: Nonempty hidden layer widths. A value of ``-1`` requires an explicit input dimension.
            input_dim: Input feature width. None infers it from the first input's last dimension.
            output_dim: Output shape. None returns the last hidden layer's activated features.
            activation: Activation function.
            last_activation: Activation appended at the end, without replacing any hidden-layer activation.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        if input_dim is None and -1 in hidden_dims:
            raise ValueError("hidden_dims containing -1 require an explicit input_dim.")

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        if input_dim is None:
            layers.append(nn.LazyLinear(hidden_dims_processed[0]))
        else:
            layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        elif output_dim is not None:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float, ...]) -> None:
        """Initialize the weights of the MLP after any LazyLinear has been materialized by a forward pass.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return 