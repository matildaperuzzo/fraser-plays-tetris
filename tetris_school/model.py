import torch
import torch.nn as nn

from tetris_school.layers import HiddenBlock, Head


class Fraser(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, num_states: int = 3, num_actions: int = 4):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([HiddenBlock(hidden_size) for i in range(layer_number)])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.head = Head(hidden_size, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)

class Jordan(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, input_size: int = 10, num_actions: int = 4):
        super().__init__()

        self.embed = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([HiddenBlock(hidden_size) for i in range(layer_number)])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.head = Head(hidden_size, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        floor = torch.zeros(len(x),dtype=x.dtype,device=x.device)
        floor_index = torch.argwhere(x == 1).to(x.dtype)
        floor[floor_index[:,0]] = floor_index[:,1]+1
        cieling = torch.zeros_like(floor)
        cieling_index = torch.argwhere(x == 2).to(x.dtype)
        cieling[cieling_index[:,0]] = cieling_index[:,1]+1
        x = torch.cat([floor,cieling]).float()
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)