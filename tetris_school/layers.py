import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()

        # self.self_attn = nn.MultiheadAttention(
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        # )
        # self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x  # self-attention block
        # x = self.self_attn_norm(x)
        # x, _ = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     need_weights=False,
        # )
        # x = residual + x
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

        return self.toaction(x)
