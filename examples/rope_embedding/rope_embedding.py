import torch
from scripts.kernel.rope_embedding import fast_rope_embedding
from scripts.kernel.rope_embedding import inplace_rope_embedding

batch_size = 2
seq_len = 3
n_heads = 4
head_dim = 5
Q = torch.randn(batch_size, seq_len, n_heads, head_dim)
cos = torch.randn(seq_len, head_dim)
sin = torch.randn(seq_len, head_dim)
position_ids = torch.LongTensor([[0, 1, 2, 1], [1, 2, 0, 2]])
breakpoint()