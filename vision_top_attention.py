"""Core vision topological attention modules.
Topologically-induced modulation of the regular attention matrix, the so-called
2-level block Toeplitz masking mechanism, was introduced in this paper:
https://arxiv.org/abs/2107.07999.
"""

import abc
from collections.abc import Iterable  # pylint: disable=g-importing-member
import functools
from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat
import numpy as np
from typing import Optional, Callable

def topological_dot_product_attention_weights(
    query,
    key,
    toeplitz_params,
    bias = None,
    mask = None,
    dropout_rate = 0.,
    nb_x_patches = 0,
    nb_y_patches = 0):
  """Computes dot-product attention weights given query and key.
  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.
  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    toeplitz_params: tensor defining parameters of the 2d-level block Toeplitz
      mask.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    dropout_rate: dropout rate
    nb_x_patches: number of patches in a fixed column.
    nb_y_patches: number of patches in a fixed row.
  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], ('q, k batch dims must match.')
  assert query.shape[-2] == key.shape[-2], ('q, k num_heads must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / math.sqrt(depth)

  query = F.relu(query) + 1e-8
  key = F.relu(key) + 1e-8

  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = torch.einsum(
      '...qhd,...khd->...hqk', query, key)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    attn_weights =  attn_weights.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

  grid_i = torch.arange(nb_x_patches)
  grid_j = torch.arange(nb_y_patches)
  dist_index = (grid_i[:, None, None, None] - grid_i[None, None, :, None] +
                nb_x_patches) * 2 * nb_y_patches + grid_j[
                    None, :, None, None] - grid_j[None, None,
                                                  None, :] + nb_y_patches

  dist_index = dist_index.reshape((-1,))
  toeplitz_mask = repeat(toeplitz_params, 'h w -> h w c', c=toeplitz_params.shape[-1]) 
  toeplitz_mask = toeplitz_mask.reshape(toeplitz_params.shape[0],-1)[:,dist_index,]
  
  toeplitz_mask = toeplitz_mask.reshape(
      (toeplitz_params.shape[0], nb_x_patches * nb_y_patches,
       nb_x_patches * nb_y_patches))
  toeplitz_mask = F.pad(toeplitz_mask, (0, 0, 1, 0, 0, 0), value=1.0) #needed to add in the CLS token 
  toeplitz_mask = F.pad(toeplitz_mask, (1, 0, 0, 0, 0, 0), value=1.0) #needed for the CLS token in VIT
  toeplitz_mask = toeplitz_mask[None, Ellipsis]
  attn_weights = attn_weights * torch.abs(toeplitz_mask) + 1e-8

  # normalize the attention weights
  attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
  # apply attention dropout
  attn_weights = F.dropout(attn_weights)

  return attn_weights

def dot_product_attention_2d_vis(query,
                          key,
                          value,
                          toeplitz_params,
                          bias = None,
                          mask = None,
                          dropout_rate = 0.,
                          nb_x_patches = 0,
                          nb_y_patches = 0):
  """Computes dot-product attention given query, key, and value.
  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.
  Note: query, key, value needn't have any batch dimensions.
  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    toeplitz_params: tensor defining parameters of the 2d-level block Toeplitz
      mask.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    dropout_rate: dropout rate
    nb_x_patches: number of patches in a fixed column.
    nb_y_patches: number of patches in a fixed row.
  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = topological_dot_product_attention_weights(
      query, key, toeplitz_params, bias, mask,
      dropout_rate,  nb_x_patches, nb_y_patches)

  # return weighted sum over values for each query position
  return torch.einsum(
      '...hqk,...khd->...qhd', attn_weights, value)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.
    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      param_dtype: the dtype passed to parameter initializers (default:
        float32).
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      nb_x_patches: number of patches in a fixed column,
      nb_y_patches: number of patches in a fixed row,
  """
  def __init__(self, num_heads = 0, 
        qkv_features = None,
        out_features = None, 
        dropout_rate: float = 0.,
        attention_fn = dot_product_attention,
        nb_x_patches = 0,
        nb_y_patches = 0,
            ):
    super().__init__()

  def forward(self,
               inputs_q,
               inputs_kv,
               mask = None,
               ):
    """Applies multi-head dot product attention on the input data.
    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.
    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        nn.Linear,
        out_features=self.num_heads*head_dim
        )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q.shape[-1])(inputs_q),
                         dense(inputs_kv.shape[-1])(inputs_kv),
                         dense(inputs_kv.shape[-1])(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    #TODO: ADD DECODING
  

    toeplitz_params = nn.Parameter(torch.ones(
        query.shape[-2],
        4 * self.nb_x_patches * self.nb_y_patches, #must be a multiple of 4 for the patch dims
    ))

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        toeplitz_params,
        mask=mask,
        dropout_rate=self.dropout_rate,
        nb_x_patches=self.nb_x_patches,
        nb_y_patches=self.nb_y_patches)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = dense(x.shape[-1])(x)
    return out.reshape(inputs_q.shape[0], -1, self.num_heads, head_dim )

#TODO: Straightforward to plug into ViT https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
#create a simplified version of TopVit https://github.com/google-research/google-research/blob/0075db7f5a2ca694dbd3ff0717778b2da22c2cea/topological_transformer/images/topvit.py

