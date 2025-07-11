import math
from torch.nn.functional import linear
from torch.nn.functional import (
    _mha_shape_check,
    _canonical_mask,
    _none_or_dtype,
    _in_projection_packed,
)
import torch
from typing import Optional

# Efficient implementation equivalent to the following:
# Attention is available in op_set 23, But currently no impl has this support
def scaled_dot_product_attention_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    num_heads,
    head_dim,
) -> torch.Tensor:
    query = query.view(-1, num_heads, head_dim).transpose(0, 1) # 170, 1, 512 -> 170, 16, 32 -> 16, 170, 32
    key = key.view(-1, num_heads, head_dim).permute(1,2,0)  # 170, 1, 512 -> 170, 16, 32 -> 16, 170, 32 ->16, 32, 170
    value = value.view(-1, num_heads, head_dim).transpose(0, 1) # 170, 1, 512 -> 170, 16, 32 -> 16, 170, 32

    scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    attn_bias = torch.zeros(
        1, query.shape[1], query.shape[1], dtype=query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            # attn_mask[attn_mask != float("-inf")] = 0
            # attn_mask[attn_mask == float("-inf")] = 1
            attn_weight += attn_mask

    return attn_weight @ value

def multi_head_attention_forward_patched(
    query,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p: float,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
    k_cache=None,
    v_cache=None,
    first_infer=True,
):
    """
    Modified multi-head attention to externalize KV cache handling.
    Args:
        ... (same as original)
        k_cache: Cached key tensor for each layer (shape: [src_len, bsz * num_heads, head_dim]).
        v_cache: Cached value tensor for each layer (shape: [src_len, bsz * num_heads, head_dim]).
        first_infer: Boolean indicating if this is the first inference step.
    Returns:
        attn_output: Attention output.
        attn_output_weights: Attention weights (if need_weights=True).
        k: Updated key tensor.
        v: Updated value tensor.
    """
    _, _, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    q, k, v  = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim= -1)
    # proj_qkv = proj_qkv.unflatten(-1, (3, query.size(-1))).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    # = proj_qkv[0], proj_qkv[1], proj_qkv[2]

    # Update key and value with cache
    if k_cache is not None and v_cache is not None and not first_infer:
        k = torch.cat([k_cache, k], dim=0)
        v = torch.cat([v_cache, v], dim=0)

    attn = scaled_dot_product_attention_custom(q, k, v, attn_mask, num_heads, head_dim)

    attn = attn.permute(1, 0, 2).contiguous().view(-1, embed_dim)
    attn = linear(attn, out_proj_weight, out_proj_bias)
    attn = attn.view(-1, 1, attn.size(1))
    return attn, k, v