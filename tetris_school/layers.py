import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x

class Head(nn.Module):
    def __init__(self, embed_dim: int, num_actions: int):
        super().__init__()

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.toaction = nn.Linear(embed_dim, num_actions)

    def forward(self, x: torch.Tensor) -> dict:

        x = self.fc(x)
        x = F.gelu(x)
        x = self.norm(x)

        representations = self.toaction(x)
        if representations.ndim == 1:
            representations = representations.unsqueeze(0).unsqueeze(1)
        return {'representations': representations, 'logits': representations.mean(axis=(0,1))}
    
class HeadSimple(nn.Module):
    def __init__(self, embed_dim: int, num_actions: int):
        super().__init__()

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.toaction = nn.Linear(embed_dim, num_actions)

    def forward(self, x: torch.Tensor) -> dict:

        x = self.fc(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.toaction(x)

        return x