import torch
import torch.nn as nn
import torch.nn.functional as F
from tetris_school.layers import HiddenBlock, Head


class Fraser(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, kernel_size: tuple = (5, 5), num_actions: int = 4):
        super().__init__()

        self.kernel_size = kernel_size
        self.input_size = kernel_size[0] * kernel_size[1]

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
        batch_size = x.size(0)
        x = F.pad(x, (self.kernel_size[1] // 2, self.kernel_size[1] // 2, self.kernel_size[0] // 2, self.kernel_size[0] // 2), value=-1)

        input = torch.zeros((batch_size, *self.kernel_size), device=x.device)
        batch_idx, width_idx, height_idx = torch.where(x == 3)

        for n, i, j in zip(batch_idx, width_idx, height_idx):
            patch = x[
                n,
                i - self.kernel_size[0] // 2 : i + self.kernel_size[0] // 2 + 1,
                j - self.kernel_size[1] // 2 : j + self.kernel_size[1] // 2 + 1,
            ]

            input[n] = patch

        return input.view(x.size(0), -1).float()


class Jordan(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, input_size: tuple = (5, 5), num_actions: int = 4):
        super().__init__()

        self.input_size = 2 * input_size[0]
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
        height_range = torch.arange(x.size(2)).to(x.device)

        floor = (height_range * (x == 1)).argmax(dim=-1)
        cieling = (height_range * (x == 2)).argmax(dim=-1)

        x = torch.cat([floor, cieling], dim=-1)
        return x.float()
