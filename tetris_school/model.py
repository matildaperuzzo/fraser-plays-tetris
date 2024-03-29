import torch
import torch.nn as nn
from tetris_school.layers import HiddenBlock, Head, HeadSimple
import torch.nn.functional as F


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
        self.head = HeadSimple(hidden_size, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)

class Anderson(nn.Module):
# CNN network
    def __init__(self, hidden_size: int, layer_number: int, tot_size: int = 10, num_actions: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1) for i in range(layer_number)])
        self.classifier = nn.Linear(hidden_size*tot_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.unsqueeze(1)
        x = self.embed(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.classifier(x))
        x = self.output(x)
        return x