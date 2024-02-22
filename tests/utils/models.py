from collections import OrderedDict
from typing import Optional

import torch


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_relu: bool = True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        if apply_relu:
            self.relu = torch.nn.ReLU()

    def forward(self, hidden_states, scale_factor=1.0):
        add = self.linear(hidden_states)
        if hasattr(self, "relu"):
            add = self.relu(add)
        return add * scale_factor + hidden_states


class DummyModel(torch.nn.Module):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        hidden_dims: tuple = (64, 64, 64, 64),
        dtype: torch.dtype = torch.float32,
        pass_as_kwargs: bool = False,
    ):
        assert len(hidden_dims) >= 2, "hidden_dims must have at least 2 elements"
        super().__init__()
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        layers = torch.nn.ModuleDict()
        for i in range(len(hidden_dims) - 1):
            with_relu = i < len(hidden_dims) - 2
            layers[f"block_{i}"] = ResidualBlock(
                hidden_dims[i], hidden_dims[i + 1], with_relu
            )
        self.layers = layers
        self.pass_as_kwargs = pass_as_kwargs
        self.input_dim = hidden_dims[0]
        self.output_dim = hidden_dims[-1]
        self.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, embedding, scale=1.0):
        hidden_states = embedding
        for block in self.layers.values():
            if not self.pass_as_kwargs:
                hidden_states = block(hidden_states)
            else:
                hidden_states = block(hidden_states=hidden_states, scale_factor=scale)
        return hidden_states * scale
