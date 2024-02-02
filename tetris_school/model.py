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

