import math
import torch
from torch import nn
from torch.nn import functional as F
from module import commons
from module.modules import LayerNorm
from typing import Optional

class Encoder(nn.Module):
    """Transformer encoder with multi-head attention and feed-forward layers."""
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(
                hidden_channels,
                hidden_channels,
                n_heads,
                p_dropout=p_dropout,
                window_size=window_size,
            ) for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            FFN(
                hidden_channels,
                hidden_channels,
                filter_channels,
                kernel_size,
                p_dropout=p_dropout,
            ) for _ in range(n_layers)
        ])
        self.norm_layers_2 = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            residual = x
            x = norm1(residual + attn(x, x, attn_mask))
            residual = x
            x = norm2(residual + ffn(x, x_mask))
        return x * x_mask

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative positional embeddings."""
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0, f"Channels ({channels}) must be divisible by n_heads ({n_heads})"

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, _ = self.attention(q, k, v, attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask: Optional[torch.Tensor] = None):
        b, d, t_s  = (*key.size(),)
        # Reshape directly to avoid multiple view operations
        query = query.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)

        scores = torch.matmul(query * (self.k_channels ** -0.5), key.transpose(-2, -1))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(query * (self.k_channels ** -0.5), key_relative_embeddings.unsqueeze(0).transpose(-2, -1))
            scores = scores + self._relative_position_to_absolute_position(rel_logits)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e4)  # In-place operation

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output.add_(torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0)))

        output = output.transpose(2, 3).contiguous().view(b, d, -1)
        return output, p_attn

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        max_rel_pos = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1

        if pad_length > 0:
            padded = F.pad(relative_embeddings, (0, 0, pad_length, pad_length))
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, 1))
        x_flat = x.view(batch, heads, -1)
        x_flat = F.pad(x_flat, (0, length - 1))
        return x_flat.view(batch, heads, length + 1, -1)[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, length - 1))
        x_flat = x.view(batch, heads, -1)
        x_flat = F.pad(x_flat, (length, 0))
        return x_flat.view(batch, heads, length, -1)[:, :, :, 1:]

class FFN(nn.Module):
    """Feed-forward network with convolutional layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = "relu",
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.activation = activation

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=self._get_padding())
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=self._get_padding())
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x * x_mask
        x = self.conv_1(x)
        x = F.gelu(x) if self.activation == "gelu" else torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        return x * x_mask

    def _get_padding(self) -> int:
        if self.kernel_size == 1:
            return 0
        return (self.kernel_size - 1) // 2 if not self.causal else self.kernel_size - 1
class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels: int = 192,
        hidden_size: int = 512,
        out_channels: int = 192,
        n_heads: int = 4,
    ):
        super().__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc: torch.Tensor, ssl_mask: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor, ge: torch.Tensor) -> torch.Tensor:
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask, attn_mask) + ssl_enc + ge
        return self.c_post(x * ssl_mask)