import torch
import triton
import triton.language as tl
from scripts.kernels.utils import calculate_settings
from config.constant import MAX_FUSED_SIZE


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi)]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x))]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        logsumexp is also stable
        Take    y =         log[sum(exp(x))]
           exp(y) =             sum(exp(x))
           exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
           exp(y) =      exp(c)*sum(exp(x - c))
               y  = log(exp(c)*sum(exp(x - c)))
               y  = c + log[sum(exp(x - c))]
        This means we can set c = max(x) to make sure
        exp(x - c) always is exp(x - max(x)).
        This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.

    Args:
        logits_ptr: Pointer to the logits tensor.
        logits_row_stride: Stride between rows in the logits tensor.
        loss_ptr: Pointer to the array where the computed losses will be stored.
        logsumexp_ptr: Pointer to the array where the logsumexp values will be stored.
        labels_ptr: Pointer to the array containing the labels.
        VOCAB_SIZE: Constant representing the size of the vocabulary.
        BLOCK_SIZE: Constant representing the block size.
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x

    Args:
        logits_ptr: Pointer to the logits tensor.
        logits_row_stride: Stride between rows in the logits tensor.
        loss_ptr: Pointer to the array where the computed losses will be stored.
        logsumexp_ptr: Pointer to the array where the logsumexp values will be stored.
        labels_ptr: Pointer to the array containing the labels.
        VOCAB_SIZE: Constant representing the size of the vocabulary.
        N_CHUNKS: Constant representing the number of chunks for processing.
        BLOCK_SIZE: Constant representing the block size.

    Notes:
        This function computes the cross-entropy loss in a chunked manner,
        where the input logits are divided into multiple chunks for processing.
        It is designed to handle large inputs efficiently by processing them
        in smaller chunks to avoid memory limitations.
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride,
    dloss_ptr,
    dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]

    Args:
        logits_ptr (_type_): _description_
        logits_row_stride (_type_): _description_
        dloss_ptr (_type_): _description_
        dloss_row_stride (_type_): _description_
        logsumexp_ptr (_type_): _description_
        labels_ptr (_type_): _description_
        VOCAB_SIZE (tl.constexpr): _description_
        BLOCK_SIZE (tl.constexpr): _description_
    """
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,  # exp(x - logsumexp) - 1
        y,  # exp(x - logsumexp)
    )

    # If y == 0: dC/dx = 0 ==> we already masked it to be = 0, so dloss = 0.
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


class FastCrossEntropyLoss(torch.autograd.Function):
    """
    A fast implementation of the cross-entropy loss function with support for both small and large vocabulary sizes.

    This implementation utilizes CUDA kernels for efficient computation and is optimized for performance.

    Args:
        logits (Tensor): Logits tensor of shape (n_rows, vocab_size), where n_rows is the number of examples and
                         vocab_size is the size of the vocabulary.
        labels (Tensor): Tensor containing the true labels for each example. It has the same shape as logits.

    Returns:
        Tensor: Computed cross-entropy losses for each example.

    Note:
        This implementation handles both small and large vocabulary sizes efficiently:
        - For small vocabularies (<= 2 ** 16), it uses a single CUDA kernel (_cross_entropy_forward).
        - For large vocabularies (> 2 ** 16), it divides the computation into chunks and utilizes another CUDA kernel
          (_chunked_cross_entropy_forward) for processing.
        - It then computes the logsumexp values for each chunk and aggregates them to obtain the overall logsumexp.
        - The backward pass (_cross_entropy_backward) computes gradients efficiently using CUDA kernels.

    Examples:
        >>> logits = torch.randn(64, 1000, device='cuda')
        >>> labels = torch.randint(0, 1000, (64,), device='cuda')
        >>> loss = FastCrossEntropyLoss.apply(logits, labels)
    """

    @staticmethod
    def forward(ctx, logits, labels):
        """
        Computes the forward pass of the cross-entropy loss function.

        Args:
            logits (Tensor): Logits tensor of shape (n_rows, vocab_size), where n_rows is the number of examples and
                             vocab_size is the size of the vocabulary.
            labels (Tensor): Tensor containing the true labels for each example. It has the same shape as logits.

        Returns:
            Tensor: Computed cross-entropy losses for each example.

        Note:
            This implementation handles both small and large vocabulary sizes efficiently:
            - For small vocabularies (<= 2 ** 16), it uses a single CUDA kernel (_cross_entropy_forward).
            - For large vocabularies (> 2 ** 16), it divides the computation into chunks and utilizes another CUDA kernel
              (_chunked_cross_entropy_forward) for processing.
            - It then computes the logsumexp values for each chunk and aggregates them to obtain the overall logsumexp.
        """
        n_rows, vocab_size = logits.shape
        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda")
        if n_chunks == 1:
            # for the small vocabs <= 2 ** 16 like llama and Mistral
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device="cuda")
            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE = vocab_size,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps  = num_warps,
            )
        else:
            # For large vocabs > 2 ** 16 like Gemma 256K
            logsumexp = torch.empty(
                (
                    n_rows,
                    n_chunks,
                ),
                dtype=torch.float32,
                device="cuda",
            )
            _chunked_cross_entropy_forward[
                (
                    n_rows,
                    n_chunks,
                )
            ](
                logits,
                logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                num_warps=32,
            )
            logsumexp = torch.logsumexp(logsumexp, dim=1)  # Row sum
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)  # Don't forget to mask padding out!
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        """
        Computes the backward pass of the cross-entropy loss function.

        Args:
            dlosses (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tensor: Gradient of the loss with respect to the logits.

        Note:
            This implementation efficiently computes gradients using CUDA kernels and supports both small and large
            vocabulary sizes.
        """
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)
        _cross_entropy_backward[
            (
                n_rows,
                n_blocks,
            )
        ](
            logits,
            logits.stride(0),
            dlosses,
            dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        return logits, None, None


def fast_cross_entropy_loss(logits, labels):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)

    loss = FastCrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items

if __name__ == "__main__":    
    batch_size = 32
    seq_len = 10
    vocab_size = 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = fast_cross_entropy_loss(logits, labels)
    print("Cross-entropy loss:", loss.item())
    