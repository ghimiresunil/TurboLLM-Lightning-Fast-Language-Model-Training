import triton
from config.constant import MAX_FUSED_SIZE

next_power_of_2 = triton.next_power_of_2

def calculate_settings(n: int):
    """Calculate optimal settings for Triton kernel based on input size.

    Args:
        n (int): Input size for Triton kernel.

    Returns:
        tuple: A tuple containing the BLOCK_SIZE and num_warps calculated based on input size.
            BLOCK_SIZE (int): The calculated block size.
            num_warps (int): The number of warps to use based on the block size.

    Raises:
        RuntimeError: If the input size exceeds the maximum CUDA blocksize.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
