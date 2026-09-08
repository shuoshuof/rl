from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class NetworkCfg:
    """Base configuration for a model's complete network structure.

    Architecture-specific subclasses group the component configurations consumed by that model. An actor may, for
    example, combine encoder, transformer, and MLP component configurations in one structure.

    Example:

    .. code-block:: python

        @configclass
        class ActorNetworkCfg(NetworkCfg):
            mlp: MLPCfg = MLPCfg(
                hidden_dims=[512, 128, 64],
                activation="elu",
            )
    """

    pass


@configclass
class MLPCfg:
    """Configuration for one multi-layer perceptron component."""

    hidden_dims: list[int] = MISSING
    """Output dimensions of the hidden layers, in execution order."""

    activation: str = "elu"
    """Activation function applied after each hidden linear layer."""
