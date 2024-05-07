import torch
from scripts.kernel.rope_embedding import fast_rope_embedding
from scripts.kernel.rope_embedding import inplace_rope_embedding

def run_fast_rope_embedding():
    # Define input tensors
    batch_size = 2
    seq_len = 3
    n_heads = 3
    head_dim = 4
    
    # Define positional encoding tensors with the correct shape and move them to GPU
    cos = torch.randn(seq_len, head_dim).cuda()      # [seq_len, head_dim]
    sin = torch.randn(seq_len, head_dim).cuda()      # [seq_len, head_dim]

    Q = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda()  # [batch_size, seq_len, n_heads, head_dim]
    
    # Test fast implementation
    Q_fast, _ = fast_rope_embedding(Q, Q, cos, sin)
    print("Fast implementation output:", Q_fast.shape)

def run_inplace_rope_embedding():
    # Define input tensors
    Q = torch.randn(2, 3, 4, 5)  # Example shape, replace with your actual data
    K = torch.randn(2, 3, 4, 5)  # Example shape, replace with your actual data
    cos = torch.randn(1, 1, 4, 5)  # Example shape, replace with your actual data
    sin = torch.randn(1, 1, 4, 5)  # Example shape, replace with your actual data
    position_ids = torch.LongTensor([[0, 1, 2, 1], [1, 2, 0, 2]])  # Example indices, replace with your actual data

    # Call inplace_rope_embedding function
    transformed_Q, transformed_K = inplace_rope_embedding(Q, K, cos, sin, position_ids)

    # Use transformed_Q and transformed_K as per your requirements
    print(transformed_Q.shape)
    print(transformed_K.shape)

if __name__ == "__main__":
    run_fast_rope_embedding()
    run_inplace_rope_embedding()
