from typing import override

import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.W_query: nn.Parameter = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key: nn.Parameter = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value: nn.Parameter = nn.Parameter(torch.randn(d_in, d_out))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        d_k = keys.shape[-1]

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)  # pyright: ignore[reportAny]
        context_vec = attn_weights @ values
        return context_vec
