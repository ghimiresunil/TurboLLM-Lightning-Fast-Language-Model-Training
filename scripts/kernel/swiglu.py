import torch
import triton
import triton.language as tl

@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Forward pass of the SwiGLU activation function.
    
    Args:
        e (Tensor): Input tensor.
        g (Tensor): Input tensor.
        h (Tensor): Output tensor.
        n_elements (int): Total number of elements in the input tensor.
        BLOCK_SIZE (int): Block size for parallelization.

    Returns:
        None
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)#.to(tl.float32)
    
    # Forward pass of SwiGLU activation function
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    
    h_row = f_row * g_row

    # Store the result in the output tensor
    tl.store(h + offsets, h_row, mask=mask)
    
def swiglu_fg_kernel(e, g):
    """
    Forward pass of SwiGLU activation function for a batch of inputs.

    Args:
        e (Tensor): Input tensor.
        g (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after applying the SwiGLU activation function.
    """
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE=1024,)
    return h

@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Backward pass of the SwiGLU activation function.
        e = e.float()
        se = 1.0 / (1.0 + torch.exp(-e))
        f = (se * e).to(dtype)
        h = f * g
        df = DW * f
        dg = DW * g
        de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    Args:
        DW (Tensor): Derivative tensor.
        e (Tensor): Input tensor.
        g (Tensor): Input tensor.
        n_elements (int): Total number of elements in the input tensor.
        BLOCK_SIZE (int): Block size for parallelization.

    Returns:
        None
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    DW_row = tl.load(DW + offsets, mask=mask, other=0)#.to(tl.float32)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)#.to(tl.float32)
    
    # Backward pass of SwiGLU activation function
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row, mask=mask) # h = f * g
    tl.store(e + offsets, df_row, mask=mask) # df = DW * f
    tl.store(g + offsets, de_row, mask=mask) # de
    
def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    """
    Backward pass of SwiGLU activation function for a batch of inputs.

    Args:
        DW (Tensor): Derivative tensor.
        e (Tensor): Input tensor.
        g (Tensor): Input tensor.

    Returns:
        tuple: Derivative tensor and updated input tensors.
    """
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024,)
    return DW, e, g
