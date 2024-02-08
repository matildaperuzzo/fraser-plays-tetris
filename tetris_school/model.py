import torch
import torch.nn as nn

from tetris_school.layers import HiddenBlock, Head


class Fraser(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, input_size: tuple = (5, 5), num_actions: int = 4):
        super().__init__()

        self.input_size = input_size[0] * input_size[1]
        self.embed = nn.Linear(self.input_size, hidden_size)
        self.layers = nn.ModuleList([HiddenBlock(hidden_size) for i in range(layer_number)])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = Head(hidden_size, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.state_transform(x)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)

    def state_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).float()


class Jordan(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, input_size: tuple = (5, 5), num_actions: int = 4):
        super().__init__()

        self.input_size = 2 * input_size[0]
        self.embed = nn.Linear(self.input_size, hidden_size)
        self.layers = nn.ModuleList([HiddenBlock(hidden_size) for i in range(layer_number)])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = Head(hidden_size, num_actions=num_actions)

        # store constant height range
        self.register_buffer("height_range", torch.arange(input_size[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.state_transform(x)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)

    def state_transform(self, x: torch.Tensor) -> torch.Tensor:

        floor = (self.height_range * (x == 1)).argmax(dim=-1)
        cieling = (self.height_range * (x == 2)).argmax(dim=-1)

        x = torch.cat([floor, cieling], dim=-1)
        return x.float()
