import triton
import torch
import triton.language as tl
from scripts.kernel.utils import calculate_settings
from config.constant import ROPE_GROUP_SIZE

@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        
    Args:
        Q (torch.Tensor): Tensor containing the input sequence embeddings.
        Q_row_stride (int): Row stride of the input tensor Q.
        cos (torch.Tensor): Tensor containing the cosine positional encoding.
        cos_row_stride (int): Row stride of the cosine positional encoding tensor.
        sin (torch.Tensor): Tensor containing the sine positional encoding.
        sin_row_stride (int): Row stride of the sine positional encoding tensor.
        seqlen (int): Length of the sequence.
        head_dim (int): Dimensionality of each attention head.
        n_heads (int): Number of attention heads.
        BACKWARD_PASS (bool): Flag indicating whether it's a backward pass or not.
        BLOCK_SIZE (int): Size of the computation block
    """
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim
    
    sin1 = tl.load(sin + (row_position % seqlen) * sin_row_stride + half_head_dim*0 + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % seqlen) * cos_row_stride + half_head_dim*0 + col_offsets, mask=mask, other=0)
    
    if BACKWARD_PASS:
        sin1 = -sin1
    
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)
    
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position + Q_row_stride + k + head_dim + col_offsets + half_head_dim
        
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)

class FastRopeEmbedding(torch.autograd.Function):
    """Custom autograd function for fast implementation of Rope Embedding.

    This function performs efficient Rope Embedding operations using CUDA kernels.
    
    Args:
        Q (torch.Tensor): Query tensor.
        cos (torch.Tensor): Cosine tensor.
        sin (torch.Tensor): Sine tensor.
    
    Returns:
        torch.Tensor: Transformed query tensor.

    Example:
        >>> Q = torch.randn(2, 3, 4, 5)
        >>> cos = torch.randn(3, 4)
        >>> sin = torch.randn(3, 4)
        >>> result = FastRopeEmbedding.apply(Q, cos, sin)
    """
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])
    
        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        
        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)
        _rope_embedding[(n_rows, n_groups, )](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin 
        return Q.view(batch, seq_len, n_heads, head_dim)
    
    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        # Must be reshape not view
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        _rope_embedding[(n_rows, ctx.n_groups, )](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None
    

class SlowRopeEmbedding(torch.autograd.Function):
    """Custom autograd function for slow implementation of Rope Embedding.

    This function performs Rope Embedding operations without CUDA kernels.

    Args:
        Q (torch.Tensor): Query tensor.
        cos (torch.Tensor): Cosine tensor.
        sin (torch.Tensor): Sine tensor.
        position_ids (torch.Tensor): Position indices.

    Returns:
        torch.Tensor: Transformed query tensor.

    Example:
        >>> Q = torch.randn(2, 3, 4, 5)
        >>> cos = torch.randn(1, 1, 4, 5)
        >>> sin = torch.randn(1, 1, 4, 5)
        >>> position_ids = torch.LongTensor([[0, 1, 2, 1], [1, 2, 0, 2]])
        >>> result = SlowRopeEmbedding.apply(Q, cos, sin, position_ids)
    """
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        
        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1]//2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        # RH_Q *= sin
        # Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q
    
    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1]//2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        # RH_dY *= sin
        # dY += RH_dY
        return dY, None, None, None

def fast_rope_embedding(Q, K, cos, sin):
    """Fast implementation of Rope Embedding for query and key tensors.

    Args:
        Q (torch.Tensor): Query tensor.
        K (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine tensor.
        sin (torch.Tensor): Sine tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors.
    """
    Q = FastRopeEmbedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = FastRopeEmbedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K

def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    """In-place implementation of Rope Embedding for query and key tensors.

    Args:
        Q (torch.Tensor): Query tensor.
        K (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine tensor.
        sin (torch.Tensor): Sine tensor.
        position_ids (torch.Tensor): Position indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed query and key tensors.
    """
    Q = SlowRopeEmbedding.apply(Q, cos, sin, position_ids)
    K = SlowRopeEmbedding.apply(K, cos, sin, position_ids)
    return Q, K
