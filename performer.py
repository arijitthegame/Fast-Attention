"""
Almost verbatim code from Lucidrains. With the added RPE from the 
SPE paper https://arxiv.org/abs/2105.08399
 and the FFT based Toeplitz matrix multiplication from https://proceedings.neurips.cc//paper/2021/file/c0f168ce8900fa56e57789e2a2f2c9d0-Paper.pdf.

To use the performer with various mechanisms, you need to set the following parameters:
In Attention class, set dim = heads * dim_head 
In Performer block, set dim = dim as above
And in PerformerEncoder class, n_heads=heads,dim=dim_head (from before),
d_model=n_heads*dim_head, num_realizations=dim_head,rel_pos_bins=n_heads*dim_head
"""

from functools import partial
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from spe_filter import *


# helpers


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


# token shifting helper and classes


def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return F.pad(t, (0, 0, amount, -amount), value=0.0)


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(
            map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts))
        )
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py


def softmax_kernel(
    data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.amax(data_dash, dim=-1, keepdim=True).detach()
            )
            + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
            )
            + eps
        )

    return data_dash.type_as(data)


def generalized_kernel(
    data,
    *,
    projection_matrix,
    kernel_fn=nn.ReLU(),
    kernel_epsilon=0.001,
    normalize_data=True,
    device=None,
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    """
    This code won't work unless torch >= 1.8
    """
    unstructured_block = torch.randn((cols, cols), device=device)

    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
    context = torch.einsum("...nd,...ne->...de", k, v)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
    return out


# inefficient causal linear attention, without cuda code, for reader's reference (for debugging)


def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q) + eps)
        context = torch.einsum("...nd,...ne->...nde", k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum("...nde,...nd,...n->...ne", context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            print(
                "Fix this code please. Defaulting to memory inefficient non-cuda version"
            )
            self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device,
            )
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out


# a module for keeping track of when to update the projections


class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if (
            exists(self.feature_redraw_interval)
            and self.calls_since_last_redraw >= self.feature_redraw_interval
        ):
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented


# classes


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_length,
        causal=False,
        heads=8,
        dim_head=64,
        nb_features=16,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout_prob=0.0,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True,
        use_rot_emb=False,
        use_spe=False,
        use_mask_pos=False,
        eps=1e-6,
        normalize=False,
    ):
        super().__init__()

        self.heads = heads
        self.dim = dim  # output_dim
        self.causal = causal
        self.dim_head = dim_head
        self.nb_features = nb_features
        self.feature_redraw_interval = feature_redraw_interval
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.dropout_prob = dropout_prob
        self.no_projection = no_projection
        self.qkv_bias = qkv_bias
        self.attn_out_bias = attn_out_bias
        self.use_rot_emb = use_rot_emb
        self.use_spe = use_spe
        self.use_mask_pos = use_mask_pos
        self.max_seq_length = max_seq_length
        self.eps = eps  # for numerical stability
        self.normalize = normalize

        if self.use_spe:
            self.spe = SPEFilter(gated=True, code_shape=(self.heads, self.dim_head))

        self.fast_attention = FastAttention(
            self.dim_head,
            self.nb_features,
            causal=self.causal,
            generalized_attention=self.generalized_attention,
            kernel_fn=self.kernel_fn,
            no_projection=self.no_projection,
        )
        if self.use_mask_pos:
            self.create_projection = partial(
                gaussian_orthogonal_random_matrix,
                nb_rows=self.nb_features,
                nb_columns=self.dim_head,
            )
            self.projection_matrix = self.create_projection()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads  # for all practical purposes dim == inner_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, queries, keys, values, rpe=None, **kwargs):
        """
        Must provide rpe if using spe or rot emb or mask pos.
        """

        b, n, _ = queries.shape
        h = self.heads

        q, k, v = self.to_q(queries), self.to_k(keys), self.to_v(values)
        tgt_len = k.shape[1]

        if self.use_rot_emb is True:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
            )
            q, k = apply_rotary_pos_emb(q, k, rpe)

        if self.use_spe is True:
            q, k = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q, k))
            q, k = self.spe(q, k, rpe)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)
            q, k = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k))

        if self.use_mask_pos:
            # Compute the KV matrix #computing and storing a lot of intermediate tensors
            # TODO: make this more efficient
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                device=q.device,
            )
            q, k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k))
            if self.normalize:
                q = q / (q.norm(dim=-1, keepdim=True) + 1e-10)
                k = k / (k.norm(dim=-1, keepdim=True) + 1e-10)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

            # Compute the KV matrix
            k = rearrange(k, "b h n d -> b h d n", h=h)
            v = rearrange(v, "b n (h d) -> b n h d", h=h)
            q = rearrange(q, "b h n d -> b n h d", h=h)
            kv = torch.einsum("bhdn,bnhm->bhmdn", k, v)

            # Efficient matrix multiplication
            u = torch.fft.rfft(rpe, dim=-1)  # rpe.shape = [num_heads, 2*tgt_len]

            y = torch.fft.rfft(
                kv, n=2 * tgt_len, dim=-1
            )  # KV.shape  = [bsz, num_heads, v_dim, k_dim, tgt_len]
            y = torch.einsum("hn,bhmdl->bhmdn", u, y)
            weighted_kv = torch.fft.irfft(y, dim=-1)[:, :, :, :, tgt_len:]

            y1 = torch.fft.rfft(
                k, n=2 * tgt_len, dim=-1
            )  # k.shape  = [bsz, num_heads, k_dim, tgt_len]
            y1 = torch.einsum("hn,bhdn->bhdn", u, y1)
            weighted_k = torch.fft.irfft(y1, dim=-1)[:, :, :, tgt_len:]

            # Compute the normalizer
            Z = 1 / (torch.einsum("nlhd,nhdl->nlh", q, weighted_k) + self.eps)
            Z = rearrange(
                Z, "b n h -> b h n"
            )  # transpose by keeping the batch dim fixed

            # Finally compute and return the new values
            # key_dim = query_dim = value_dim
            out = torch.einsum("bnhd,bhddn,bhn->bnhd", q, weighted_kv, Z)
            out = rearrange(out, "b n h d -> b n (h d)")
            out = self.to_out(out)

        else:
            out = self.fast_attention(q, k, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.to_out(out)

        return self.dropout(out)


# positional embeddings


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


# rotary positional embedding helpers


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, "() n (j d) -> n j d", j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# sinusoidal positional embeddings


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x):
        return self.emb[None, : x.shape[1], :].to(x)


# performer


class PerformerBlock(nn.Module):
    """
    This is the performer SELF ATTENTION block.
    """

    def __init__(self, attention, d_model, dropout=0.1, activation="relu"):
        super(PerformerBlock, self).__init__()

        d_ff = 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, rpe=None, **kwargs):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.

        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(x, x, x, rpe=rpe, **kwargs))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)


class PerformerEncoder(nn.Module):
    def __init__(
        self,
        layers,
        n_heads,
        dim,
        d_model,
        max_seq_length,
        num_realizations=1,
        norm_layer=None,
        rel_pos_bins=None,
        use_spe=False,
        spe_type=None,
        kernel_size=None,
        use_rot_emb=False,
        use_mask_pos=False,
        normalize=False,
    ):

        super(PerformerEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.dim = dim  # dim per head
        self.n_heads = n_heads
        self.d_model = d_model  # d_ model = dim per head * n_heads
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins  # num_heads * dim
        self.use_rot_emb = use_rot_emb  # use rotary positional embeddings in Rotoformer
        self.use_spe = (
            use_spe  # gated mechanism for positional embeddings using conv or sine
        )
        self.spe_type = spe_type  # conv/sine spe
        self.use_mask_pos = use_mask_pos  # fft masking via Toeplitz matrices
        self.normalize = normalize  # normalize keys/queries
        if self.use_mask_pos is True:
            self.relative_positional_bias = nn.Parameter(
                torch.randn(self.n_heads, 2 * rel_pos_bins - 1)
            )

        if self.spe_type == "sine":
            self.sine_spe = SineSPE(
                num_heads=self.n_heads,
                in_features=self.dim,
                num_sines=self.d_model,
                num_realizations=self.num_realizations,
            )
        elif self.spe_type == "conv":
            self.conv_spe = ConvSPE(
                self.n_heads, self.dim, self.d_model, self.kernel_size
            )

        if self.use_rot_emb is True:
            self.pos_emb = FixedPositionalEmbedding(self.d_model, self.max_seq_length)
            self.layer_pos_emb = FixedPositionalEmbedding(self.dim, self.max_seq_length)

    def forward(self, x, rpe=None, **kwargs):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: not compute attention for [PAD] tokens. #TODO: add this to the transformer encoder
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # We assume that the sequences have the right length and nothing is padded.
        # TODO: ADD in attention mask if there is a PAD token

        if self.use_mask_pos is True:
            if L <= self.rel_pos_bins:
                rpe = torch.cat(
                    (
                        self.relative_positional_bias[:, 0].unsqueeze(1),
                        self.relative_positional_bias[
                            :, self.rel_pos_bins - L : self.rel_pos_bins + L - 1
                        ],
                    ),
                    dim=1,
                )
            else:
                rpe = torch.cat(
                    (
                        self.relative_positional_bias[:, 0]
                        .unsqueeze(1)
                        .repeat(1, L - self.rel_pos_bins + 1),
                        self.relative_positional_bias,
                        self.relative_positional_bias[:, -1]
                        .unsqueeze(1)
                        .repeat(1, L - self.rel_pos_bins),
                    ),
                    dim=1,
                )

        elif self.use_spe is True:
            if self.spe_type == "sine":
                rpe = self.sine_spe((self.n_heads, self.max_seq_length))
            elif self.spe_type == "conv":  # conv gives poor results
                rpe = self.conv_spe(self.n_heads, self.dim)
            else:
                raise ValueError("spe_type not supported")
        else:
            # we assume that L is the max seq length
            x += self.pos_emb(x)
            rpe = self.layer_pos_emb(x)

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, rpe=rpe)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x


# TODO: Add some unit tests to show usage. @author: arijitthegame
# TODO: Add attention mask for LM experiments.
