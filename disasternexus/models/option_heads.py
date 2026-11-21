"""Task-conditioned option scoring heads for recognition tasks (DTR, DSR, BBR)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


class SingleTaskOptionHead(nn.Module):
    """Bilinear projection scoring hidden states against option text embeddings."""

    def __init__(self, lm_hidden_dim: int, head_dim: int = 256) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.hidden_norm = nn.LayerNorm(lm_hidden_dim)
        self.embed_norm = nn.LayerNorm(lm_hidden_dim)
        self.hidden_proj = nn.Linear(lm_hidden_dim, head_dim)
        self.embed_proj = nn.Linear(lm_hidden_dim, head_dim)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_state: Tensor, option_embeddings: Tensor) -> Tensor:
        # hidden_state: [D], option_embeddings: [K, D]
        h = self.hidden_proj(self.hidden_norm(hidden_state))
        e = self.embed_proj(self.embed_norm(option_embeddings))
        scale = torch.exp(self.log_scale) / math.sqrt(self.head_dim)
        return scale * torch.matmul(e, h) + self.bias


class TaskConditionedOptionHeads(nn.Module):
    """Manage one option scoring head per recognition task."""

    def __init__(
        self,
        tasks: Sequence[str],
        lm_hidden_dim: int,
        head_dim: int = 256,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            self._sanitize(task): SingleTaskOptionHead(lm_hidden_dim, head_dim)
            for task in tasks
        })

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace(" ", "_")

    def forward(
        self,
        task_name: str,
        hidden_state: Tensor,
        option_embeddings: Tensor,
    ) -> Tensor:
        key = self._sanitize(task_name)
        if key not in self.heads:
            raise KeyError(f"Unsupported recognition option-head task: {task_name}")
        return self.heads[key](hidden_state, option_embeddings)
